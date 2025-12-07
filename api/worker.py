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
)
from src.deep_evaluator import DeepEvaluator
from src.grok_client import GrokClient

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
FAST_SCREEN_DIR = DATA_DIR / "evaluations" / "fast_screen"

# Ensure directories exist
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
FAST_SCREEN_DIR.mkdir(parents=True, exist_ok=True)

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


async def fetch_x_user(handle: str) -> Optional[dict]:
    """Fetch user data from X API."""
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

    # Use httpx with OAuth1 (convert to headers manually)
    import requests

    # Use requests for OAuth1 (httpx doesn't support OAuth1 natively)
    try:
        r = requests.get(url, auth=auth, params=params, timeout=30)

        if r.status_code == 404:
            return None

        if r.status_code == 429:
            # Rate limited
            raise Exception("X API rate limited")

        r.raise_for_status()
        data = r.json()

        if "data" not in data:
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
    except Exception as e:
        print(f"[worker] Error fetching @{handle}: {e}")
        return None


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
        user_data = await fetch_x_user(handle)

        if not user_data:
            update_submission_status(
                submission_id,
                "failed",
                stage="fetching",
                error_message=f"User @{handle} not found on X",
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

    functions = [process_submission]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = get_redis_settings()

    # Job settings
    max_jobs = 2  # Process 2 jobs concurrently
    job_timeout = 300  # 5 minute timeout per job
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
