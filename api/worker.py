"""
Background worker for processing handle submissions.
Uses arq (async Redis queue) for job processing.
"""

import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from arq import cron
from arq.connections import RedisSettings
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import (
    get_submission,
    increment_submission_attempts,
    init_db,
    update_submission_status,
    update_submission_stage,
    upsert_graph_node,
    upsert_graph_edge,
)
from src.deep_evaluator import DeepEvaluator
from src.grok_client import GrokClient
from src.graph_builder import GraphBuilder
from src.ranking import compute_rankings
from src.x_client import XClient

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
FAST_SCREEN_DIR = DATA_DIR / "evaluations" / "fast_screen"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Ensure directories exist
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
FAST_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Default graph expansion settings
DEFAULT_GRAPH_SETTINGS = {
    "max_depth": 1,
    "max_followers_candidate": 50000,
    "min_followers_candidate": 50,
    "max_following_per_root": 500,
    "max_liked_tweets": 100,
    "max_root_tweets": 50,
    "max_likers_per_tweet": 100,
    "max_retweeters_per_tweet": 100,
    "max_replies_per_conversation": 50,
}

# X API base URL
X_API_BASE = "https://api.twitter.com/2"


def get_redis_settings() -> RedisSettings:
    """Get Redis settings from environment."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    # Parse Redis URL
    if redis_url.startswith("redis://"):
        redis_url = redis_url[8:]
    elif redis_url.startswith("rediss://"):
        # TLS connection (Upstash)
        redis_url = redis_url[9:]
        # For Upstash, extract host:port and password
        if "@" in redis_url:
            auth, host_port = redis_url.rsplit("@", 1)
            password = auth.split(":")[-1] if ":" in auth else auth
            host, port = host_port.split(":") if ":" in host_port else (host_port, "6379")
            return RedisSettings(
                host=host,
                port=int(port),
                password=password,
                ssl=True,
            )

    # Standard redis URL parsing
    if "@" in redis_url:
        auth, host_port = redis_url.rsplit("@", 1)
        password = auth.split(":")[-1] if ":" in auth else None
    else:
        host_port = redis_url
        password = None

    if ":" in host_port:
        host, port = host_port.split(":")
    else:
        host, port = host_port, "6379"

    return RedisSettings(
        host=host,
        port=int(port),
        password=password,
    )


class XAPIError(Exception):
    """Custom exception for X API errors with specific error types."""
    pass


async def fetch_x_user(handle: str) -> Optional[dict]:
    """Fetch user data from X API.

    Returns user data dict on success, None if user not found (404).
    Raises XAPIError for other errors (auth, rate limit, etc).
    """
    handle = handle.lower().lstrip("@")

    # Build OAuth1 signature
    auth = OAuth1(
        os.environ["X_API_KEY"],
        os.environ["X_API_KEY_SECRET"],
        os.environ["X_ACCESS_TOKEN"],
        os.environ["X_ACCESS_TOKEN_SECRET"],
    )

    url = f"{X_API_BASE}/users/by/username/{handle}"
    params = {
        "user.fields": "id,name,username,description,public_metrics,location,pinned_tweet_id",
        "expansions": "pinned_tweet_id",
        "tweet.fields": "text",
    }

    # Use requests for OAuth1 (httpx doesn't support OAuth1 natively)
    import requests

    try:
        r = requests.get(url, auth=auth, params=params, timeout=30)
        print(f"[X API] GET {url} -> {r.status_code}")

        if r.status_code == 404:
            print(f"[X API] User @{handle} not found (404)")
            return None

        if r.status_code == 401:
            print(f"[X API] Unauthorized (401) - X API credentials may be invalid")
            raise XAPIError("X API credentials are invalid or expired. Please check X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET environment variables.")

        if r.status_code == 403:
            print(f"[X API] Forbidden (403) - X API access may be restricted")
            raise XAPIError("X API access forbidden. Your X Developer account may need upgraded access or the credentials may be suspended.")

        if r.status_code == 429:
            print(f"[X API] Rate limited (429)")
            raise XAPIError("X API rate limited. Please try again later.")

        r.raise_for_status()
        data = r.json()

        if "data" not in data:
            # User might be suspended or deleted
            errors = data.get("errors", [])
            if errors:
                error_detail = errors[0].get('detail', str(errors))
                print(f"[X API] User @{handle} error: {error_detail}")
                # Check if user is suspended
                if "suspended" in error_detail.lower():
                    return None  # Treat as not found
                raise XAPIError(f"X API error: {error_detail}")
            else:
                print(f"[X API] User @{handle} no data returned: {data}")
                return None

        user = data["data"]
        metrics = user.get("public_metrics", {})

        # Extract pinned tweet text if available
        pinned_tweet = None
        if "includes" in data and "tweets" in data["includes"]:
            pinned_tweet = data["includes"]["tweets"][0].get("text")

        return {
            "user_id": user["id"],
            "handle": user["username"],
            "name": user.get("name", ""),
            "bio": user.get("description", ""),
            "location": user.get("location"),
            "followers_count": metrics.get("followers_count", 0),
            "following_count": metrics.get("following_count", 0),
            "tweet_count": metrics.get("tweet_count", 0),
            "pinned_tweet": pinned_tweet,
        }
    except XAPIError:
        raise  # Re-raise our custom errors
    except requests.exceptions.RequestException as e:
        print(f"[worker] Network error fetching @{handle}: {e}")
        raise XAPIError(f"Network error connecting to X API: {e}")
    except Exception as e:
        print(f"[worker] Unexpected error fetching @{handle}: {e}")
        raise XAPIError(f"Unexpected error: {e}")


async def process_submission(
    ctx: dict,
    submission_id: str,
    handle: str,
):
    """
    Process a handle submission through the unified pipeline.

    Pipeline stages:
    1. fetching - Fetch X profile
    2. fast_screen - Quick evaluation (gate)
    3. deep_eval - Deep evaluation (only if passed fast_screen)
    4. completed/filtered_out/failed - Terminal states

    Args:
        ctx: Worker context
        submission_id: Unique submission ID
        handle: X handle to evaluate
    """
    print(f"[worker] Processing {submission_id}: @{handle}")

    # Mark as processing
    update_submission_status(submission_id, "processing", stage="fetching")
    increment_submission_attempts(submission_id)

    try:
        # Stage 1: Fetch user from X API
        update_submission_stage(submission_id, "fetching")
        try:
            user_data = await fetch_x_user(handle)
        except XAPIError as e:
            # X API error (auth, rate limit, etc) - not a user error
            update_submission_status(
                submission_id,
                "failed",
                stage="fetching",
                error_message=str(e),
            )
            return {"success": False, "error": str(e)}

        if not user_data:
            update_submission_status(
                submission_id,
                "failed",
                stage="fetching",
                error_message=f"User @{handle} not found on X (may be suspended or deleted)",
            )
            return {"success": False, "error": "User not found"}

        print(f"[worker] Fetched @{handle}: {user_data['followers_count']} followers")

        # Stage 2: Fast screen (gate)
        update_submission_stage(submission_id, "fast_screen")
        grok = GrokClient(cache_dir=str(DATA_DIR / "evaluations"))

        fast_result = grok.fast_screen(
            handle=user_data["handle"],
            bio=user_data["bio"],
            pinned_tweet=user_data.get("pinned_tweet"),
            location=user_data.get("location"),
        )

        fast_result_dict = asdict(fast_result)
        fast_result_json = json.dumps(fast_result_dict)

        print(f"[worker] Fast screen @{handle}: pass={fast_result.pass_filter}, role={fast_result.potential_role}")

        # Save fast screen result to file
        fast_path = FAST_SCREEN_DIR / f"fast_{handle.lower()}.json"
        with open(fast_path, "w") as f:
            json.dump({
                **user_data,
                "fast_screen": fast_result_dict,
                "evaluated_at": datetime.utcnow().isoformat(),
            }, f, indent=2)

        # Check if passed fast screen - if not, filter out
        if not fast_result.pass_filter:
            update_submission_status(
                submission_id,
                "filtered_out",
                stage="fast_screen",
                fast_screen_result=fast_result_json,
                error_message=f"Did not pass fast screen: {fast_result.reason}",
            )
            return {
                "success": True,
                "handle": handle,
                "fast_screen": fast_result_dict,
                "filtered_out": True,
                "reason": fast_result.reason,
            }

        # Stage 3: Deep evaluation (only for those who passed fast screen)
        update_submission_stage(submission_id, "deep_eval")
        print(f"[worker] Running deep evaluation for @{handle}...")

        deep_evaluator = DeepEvaluator(cache_dir=str(ENRICHED_DIR))

        deep_result = deep_evaluator.evaluate(
            handle=user_data["handle"],
            bio=user_data["bio"],
            followers=user_data["followers_count"],
            use_cache=False,  # Fresh evaluation for submissions
            verbose=True,
        )

        # Convert to dict (handle dataclass)
        deep_result_dict = asdict(deep_result)

        # Add graph metrics (will be 0 for new submissions)
        deep_result_dict["pagerank_score"] = 0.0
        deep_result_dict["underratedness_score"] = 0.0
        deep_result_dict["discovered_via"] = "user_submission"

        deep_result_json = json.dumps(deep_result_dict)

        print(f"[worker] Deep eval @{handle}: score={deep_result.final_score:.1f}")

        # Stage 4: Mark complete
        update_submission_status(
            submission_id,
            "completed",
            stage="completed",
            fast_screen_result=fast_result_json,
            deep_eval_result=deep_result_json,
        )

        # Note: Cache reload happens automatically on next API request
        # In multi-process deployment, worker can't directly reload API cache

        return {
            "success": True,
            "handle": handle,
            "fast_screen": fast_result_dict,
            "deep_eval": deep_result_dict,
        }

    except Exception as e:
        print(f"[worker] Error processing @{handle}: {e}")
        import traceback
        traceback.print_exc()

        update_submission_status(
            submission_id,
            "failed",
            error_message=str(e),
        )
        return {"success": False, "error": str(e)}


async def process_seed_submission(
    ctx: dict,
    submission_id: str,
    handle: str,
    depth: int = 1,
    max_following: int = 500,
    max_candidates_to_eval: int = 50,
):
    """
    Process a seed/source submission through the full graph expansion pipeline.

    Pipeline stages:
    1. fetching - Fetch X profile for the seed
    2. graph_building - Build graph from seed's follows/retweets/replies
    3. fast_screen - Fast screen all discovered candidates
    4. deep_eval - Deep evaluation on top candidates
    5. ranking - Compute PageRank and underratedness scores
    6. completed/failed - Terminal states

    Args:
        ctx: Worker context
        submission_id: Unique submission ID
        handle: X handle to use as seed
        depth: Graph expansion depth (1-5)
        max_following: Max accounts to fetch from following list
        max_candidates_to_eval: Max candidates to deep evaluate
    """
    print(f"[worker] Processing seed submission {submission_id}: @{handle} (depth={depth})")

    # Mark as processing
    update_submission_status(submission_id, "processing", stage="fetching")
    increment_submission_attempts(submission_id)

    try:
        # Stage 1: Verify seed exists on X
        update_submission_stage(submission_id, "fetching")
        try:
            seed_data = await fetch_x_user(handle)
        except XAPIError as e:
            update_submission_status(
                submission_id,
                "failed",
                stage="fetching",
                error_message=str(e),
            )
            return {"success": False, "error": str(e)}

        if not seed_data:
            update_submission_status(
                submission_id,
                "failed",
                stage="fetching",
                error_message=f"Seed @{handle} not found on X",
            )
            return {"success": False, "error": "Seed not found"}

        print(f"[worker] Verified seed @{handle}: {seed_data['followers_count']} followers")

        # Stage 2: Build graph from seed
        update_submission_stage(submission_id, "graph_building")
        print(f"[worker] Building graph from @{handle}...")

        # Initialize X client and graph builder
        x_client = XClient(cache_dir=str(RAW_DIR))
        settings = {
            **DEFAULT_GRAPH_SETTINGS,
            "max_depth": depth,
            "max_following_per_root": max_following,
        }

        builder = GraphBuilder(
            x_client=x_client,
            roots=[handle],
            settings=settings,
        )

        # Build the graph (synchronous operation)
        builder.build_graph()

        print(f"[worker] Graph built: {len(builder.nodes)} nodes, {len(builder.edges)} edges")

        # Stage 3: Fast screen all candidates
        update_submission_stage(submission_id, "fast_screen")
        print(f"[worker] Fast screening candidates...")

        grok = GrokClient(cache_dir=str(DATA_DIR / "evaluations"))

        candidates = [
            node for node in builder.nodes.values()
            if node.is_candidate and not node.is_root
        ]

        print(f"[worker] Found {len(candidates)} candidates to screen")

        fast_screened = 0
        passed_fast_screen = []

        for i, candidate in enumerate(candidates):
            try:
                fast_result = grok.fast_screen(
                    handle=candidate.handle,
                    bio=candidate.bio,
                    pinned_tweet=None,
                    location=None,
                )

                # Update node with fast screen result
                candidate.grok_relevant = fast_result.pass_filter
                candidate.grok_role = fast_result.potential_role

                # Save to file
                fast_path = FAST_SCREEN_DIR / f"fast_{candidate.handle.lower()}.json"
                with open(fast_path, "w") as f:
                    json.dump({
                        "handle": candidate.handle,
                        "bio": candidate.bio,
                        "pass_filter": fast_result.pass_filter,
                        "potential_role": fast_result.potential_role,
                        "reason": fast_result.reason,
                        "evaluated_at": datetime.utcnow().isoformat(),
                    }, f, indent=2)

                fast_screened += 1

                if fast_result.pass_filter:
                    passed_fast_screen.append(candidate)

                if (i + 1) % 20 == 0:
                    print(f"[worker] Fast screened {i + 1}/{len(candidates)}")

            except Exception as e:
                print(f"[worker] Error fast screening @{candidate.handle}: {e}")

        print(f"[worker] Fast screened {fast_screened} candidates, {len(passed_fast_screen)} passed")

        # Stage 4: Deep evaluation on top candidates
        update_submission_stage(submission_id, "deep_eval")
        print(f"[worker] Running deep evaluation...")

        deep_evaluator = DeepEvaluator(cache_dir=str(ENRICHED_DIR))

        # Sort by followers (prioritize higher quality accounts)
        passed_fast_screen.sort(key=lambda x: x.followers_count, reverse=True)

        deep_evaluated = 0
        for candidate in passed_fast_screen[:max_candidates_to_eval]:
            try:
                deep_result = deep_evaluator.evaluate(
                    handle=candidate.handle,
                    bio=candidate.bio,
                    followers=candidate.followers_count,
                    use_cache=True,
                    verbose=False,
                )

                # Update node with deep eval result
                candidate.grok_evaluated = True
                candidate.grok_score = deep_result.final_score
                candidate.grok_reasoning = deep_result.reasoning or ""

                deep_evaluated += 1

                if deep_evaluated % 10 == 0:
                    print(f"[worker] Deep evaluated {deep_evaluated}/{min(len(passed_fast_screen), max_candidates_to_eval)}")

            except Exception as e:
                print(f"[worker] Error deep evaluating @{candidate.handle}: {e}")

        print(f"[worker] Deep evaluated {deep_evaluated} candidates")

        # Stage 5: Compute rankings
        update_submission_stage(submission_id, "ranking")
        print(f"[worker] Computing rankings...")

        nodes = compute_rankings(builder.nodes, builder.edges)

        # Stage 6: Export results
        print(f"[worker] Exporting results...")

        # Export to CSV
        seed_handle = handle.lower()
        nodes_path = PROCESSED_DIR / f"nodes_{seed_handle}.csv"
        edges_path = PROCESSED_DIR / f"edges_{seed_handle}.csv"

        builder.export_to_csv(str(nodes_path), str(edges_path))

        # Stage 7: Persist to database
        print(f"[worker] Persisting {len(builder.nodes)} nodes and {len(builder.edges)} edges to database...")

        nodes_persisted = 0
        for node_id, node in builder.nodes.items():
            try:
                # Load deep eval data if available
                deep_eval_data = None
                deep_eval_path = ENRICHED_DIR / f"deep_{node.handle.lower()}.json"
                if deep_eval_path.exists():
                    with open(deep_eval_path) as f:
                        deep_eval_data = f.read()

                node_data = {
                    "id": node_id,
                    "user_id": getattr(node, "user_id", node_id),
                    "handle": node.handle,
                    "name": node.name,
                    "bio": node.bio,
                    "followers_count": node.followers_count,
                    "following_count": node.following_count,
                    "is_seed": node.is_root,
                    "is_candidate": node.is_candidate,
                    "discovered_via": node.discovered_via,
                    "depth": getattr(node, "depth", 0),
                    "pagerank_score": getattr(node, "pagerank_score", 0.0),
                    "underratedness_score": getattr(node, "underratedness_score", 0.0),
                    "grok_relevant": node.grok_relevant,
                    "grok_role": node.grok_role,
                    "grok_evaluated": getattr(node, "grok_evaluated", False),
                    "final_score": getattr(node, "grok_score", None),
                    "deep_eval_data": deep_eval_data,
                }
                upsert_graph_node(node_data)
                nodes_persisted += 1
            except Exception as e:
                print(f"[worker] Error persisting node @{node.handle}: {e}")

        edges_persisted = 0
        for edge in builder.edges:
            try:
                edge_data = {
                    "source_id": edge.source,
                    "target_id": edge.target,
                    "weight": edge.weight,
                    "interaction_type": edge.interaction_type,
                }
                upsert_graph_edge(edge_data)
                edges_persisted += 1
            except Exception as e:
                print(f"[worker] Error persisting edge: {e}")

        print(f"[worker] Persisted {nodes_persisted} nodes and {edges_persisted} edges to database")

        # Build result summary
        total_candidates = sum(1 for n in builder.nodes.values() if n.is_candidate)
        grok_relevant = sum(1 for n in builder.nodes.values() if n.grok_relevant)
        grok_evaluated = sum(1 for n in builder.nodes.values() if n.grok_evaluated)

        result_summary = {
            "seed": handle,
            "depth": depth,
            "total_nodes": len(builder.nodes),
            "total_edges": len(builder.edges),
            "candidates": total_candidates,
            "fast_screened": fast_screened,
            "passed_fast_screen": len(passed_fast_screen),
            "deep_evaluated": grok_evaluated,
            "nodes_csv": str(nodes_path),
            "edges_csv": str(edges_path),
        }

        # Mark complete
        update_submission_status(
            submission_id,
            "completed",
            stage="completed",
            deep_eval_result=json.dumps(result_summary),
        )

        print(f"[worker] Seed submission completed: {result_summary}")

        return {
            "success": True,
            **result_summary,
        }

    except Exception as e:
        print(f"[worker] Error processing seed @{handle}: {e}")
        import traceback
        traceback.print_exc()

        update_submission_status(
            submission_id,
            "failed",
            error_message=str(e),
        )
        return {"success": False, "error": str(e)}


async def startup(ctx: dict):
    """Worker startup hook."""
    print("[worker] Starting up...")
    print(f"[worker] Data dir: {DATA_DIR}")
    print(f"[worker] Enriched dir: {ENRICHED_DIR}")
    # Initialize database (ensures tables exist)
    init_db()
    print("[worker] Database initialized")


async def shutdown(ctx: dict):
    """Worker shutdown hook."""
    print("[worker] Shutting down...")


class WorkerSettings:
    """arq worker settings."""

    functions = [process_submission, process_seed_submission]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = get_redis_settings()

    # Job settings
    max_jobs = 2  # Process 2 jobs concurrently
    job_timeout = 600  # 10 minute timeout per job (increased for seed processing)
    max_tries = 3  # Retry failed jobs up to 3 times
    retry_jobs = True

    # Health check
    health_check_interval = 30


if __name__ == "__main__":
    # For local testing
    import asyncio

    async def test():
        result = await process_submission(
            {},
            "test_123",
            "karpathy",
        )
        print(f"Result: {result}")

    asyncio.run(test())
