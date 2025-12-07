"""
FastAPI Backend for Grok Recruiter

Endpoints:
- GET /candidates - List candidates with optional NL search
- GET /candidates/{handle} - Get candidate detail
- GET /stats - Get pipeline statistics
- POST /search - Natural language search with re-ranking
"""

import base64
import hashlib
import json
import math
import os
import pickle
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import networkx as nx
import requests
from arq import create_pool
from arq.connections import RedisSettings
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from jose import JWTError, jwt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import (
    # Submission functions
    create_submission as db_create_submission,
)
from api.database import (
    get_pending_approvals as db_get_pending_approvals,
)
from api.database import (
    get_saved_count,
    get_submission_queue_position,
    init_db,
    is_candidate_saved,
    migrate_db,
    save_dm_history,
)
from api.database import (
    get_saved_handles as db_get_saved_handles,
)
from api.database import (
    get_submission as db_get_submission,
)
from api.database import (
    get_submission_by_handle as db_get_submission_by_handle,
)
from api.database import (
    save_candidate as db_save_candidate,
)
from api.database import (
    unsave_candidate as db_unsave_candidate,
)
from api.database import (
    delete_submission as db_delete_submission,
    delete_failed_submissions as db_delete_failed_submissions,
    get_all_submissions as db_get_all_submissions,
)
from api.database import (
    update_submission_approval as db_update_submission_approval,
)
from api.database import (
    update_submission_status as db_update_submission_status,
)
from api.database import (
    # Graph functions
    add_graph_edge,
    get_all_graph_edges,
    get_all_graph_nodes,
    get_graph_node_by_handle,
    get_graph_stats,
    get_seed_nodes,
    update_node_pagerank,
    update_node_grok_result,
    upsert_graph_node,
    get_evaluated_candidates,
    get_relevant_candidates,
)
from api.models import (
    AddSourceRequest,
    CandidateDeepEval,
    CandidateDetailResponse,
    CandidateListItem,
    DMGenerateRequest,
    DMGenerateResponse,
    FilterRequest,
    GraphEdge,
    GraphNode,
    GraphResponse,
    GraphStatusResponse,
    HandleSubmitRequest,
    PendingApproval,
    SaveCandidateRequest,
    SavedCandidateResponse,
    SavedCandidatesListResponse,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    SeedSubmitRequest,
    StatsResponse,
    SubmissionStatus,
)
from src.grok_client import GrokClient
from src.x_client import XClient

load_dotenv()

app = FastAPI(
    title="Grok Underrated Recruiter API",
    description="Talent discovery from X graph + Grok evaluation",
    version="1.0.0",
)


# Global Redis pool for arq job queue
arq_pool = None


def get_redis_settings() -> RedisSettings:
    """Parse REDIS_URL into arq RedisSettings."""
    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        return None

    # Parse rediss:// (TLS) or redis:// URL
    if redis_url.startswith("rediss://"):
        redis_url = redis_url[9:]
        ssl = True
    elif redis_url.startswith("redis://"):
        redis_url = redis_url[8:]
        ssl = False
    else:
        return None

    # Parse auth@host:port
    if "@" in redis_url:
        auth, host_port = redis_url.rsplit("@", 1)
        password = auth.split(":")[-1] if ":" in auth else auth
    else:
        host_port = redis_url
        password = None

    if ":" in host_port:
        host, port = host_port.split(":")
    else:
        host, port = host_port, "6379"

    return RedisSettings(host=host, port=int(port), password=password, ssl=ssl)


@app.on_event("startup")
async def startup_event():
    """Initialize database, Redis pool, and load graph on startup."""
    global arq_pool
    init_db()
    migrate_db()  # Run database migrations
    load_graph_from_pickle()

    # Initialize arq Redis pool if REDIS_URL is configured
    redis_settings = get_redis_settings()
    if redis_settings:
        try:
            arq_pool = await create_pool(redis_settings)
            print("[startup] arq Redis pool initialized")
        except Exception as e:
            print(f"[startup] Failed to connect to Redis: {e}")
            arq_pool = None
    else:
        print("[startup] No REDIS_URL configured, using BackgroundTasks fallback")


# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (Vercel preview URLs are dynamic)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- X OAuth 2.0 Configuration ---
X_CLIENT_ID = os.getenv("X_CLIENT_SECRET_ID", "")  # X uses "Client ID" naming
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET", "")
X_CALLBACK_URL = os.getenv(
    "X_CALLBACK_URL", "https://grok-recruiter-api.fly.dev/auth/x/callback"
)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://grok-underrated-recruiter.vercel.app")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # 1 week

# In-memory fallback for OAuth state (Redis preferred)
oauth_states: Dict[str, dict] = {}

# Redis client for shared OAuth state storage
import redis as redis_lib

_redis_client = None


def get_redis_client():
    """Get or create Redis client for OAuth state storage."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        return None

    try:
        _redis_client = redis_lib.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        print(f"[auth] Redis connection failed: {e}")
        return None


def store_oauth_state(state: str, data: dict, ttl: int = 600):
    """Store OAuth state in Redis (or fallback to memory)."""
    redis = get_redis_client()
    if redis:
        redis.setex(f"oauth:{state}", ttl, json.dumps(data))
    else:
        oauth_states[state] = data


def get_oauth_state(state: str) -> Optional[dict]:
    """Get OAuth state from Redis (or fallback to memory)."""
    redis = get_redis_client()
    if redis:
        data = redis.get(f"oauth:{state}")
        if data:
            redis.delete(f"oauth:{state}")  # One-time use
            return json.loads(data)
        return None
    else:
        return oauth_states.pop(state, None)


# Admin handles that can approve/reject submissions
ADMIN_HANDLES = [
    h.strip().lower() for h in os.getenv("ADMIN_HANDLES", "huwng_tran,ellenjxu_").split(",")
]


def create_jwt_token(user_data: dict) -> str:
    """Create a JWT token for authenticated user."""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        **user_data,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(authorization: str = Header(None)) -> Optional[dict]:
    """Dependency to get current authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    return verify_jwt_token(token)


async def require_auth(authorization: str = Header(...)) -> dict:
    """Dependency that requires authentication."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    user = verify_jwt_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    """Dependency that requires admin access."""
    if user.get("handle", "").lower() not in ADMIN_HANDLES:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
PROCESSED_DIR = DATA_DIR / "processed"
PROJECT_ROOT = Path(__file__).parent.parent
GRAPH_PICKLE_PATH = PROJECT_ROOT / "graph_export" / "graph.pickle"

# --- Graph State Management ---

graph_state: Dict[str, Any] = {
    "graph": nx.DiGraph(),
    "nodes": {},  # node_id -> node_data
    "pagerank": {},  # node_id -> score
    "filter_results": {},  # node_id -> FastScreenResult
    "seeds": set(),  # seed node IDs
    "loading": False,
    "last_error": None,
}

# Lazy-initialized clients
_x_client = None
_grok_client = None


def get_x_client() -> XClient:
    """Get or create X API client."""
    global _x_client
    if _x_client is None:
        cache_dir = PROJECT_ROOT / "data" / "raw"
        _x_client = XClient(cache_dir=str(cache_dir))
    return _x_client


def get_grok_client() -> GrokClient:
    """Get or create Grok API client."""
    global _grok_client
    if _grok_client is None:
        cache_dir = PROJECT_ROOT / "data" / "evaluations"
        _grok_client = GrokClient(cache_dir=str(cache_dir))
    return _grok_client


def compute_pagerank_scores():
    """Compute PageRank scores with seed personalization."""
    G = graph_state["graph"]
    seeds = graph_state["seeds"]

    if G.number_of_nodes() == 0:
        return

    # Build personalization vector
    personalization = None
    if seeds:
        valid_seeds = seeds & set(G.nodes())
        if valid_seeds:
            personalization = {uid: 1.0 / len(valid_seeds) for uid in valid_seeds}

    # Compute PageRank
    try:
        pr = nx.pagerank(
            G, alpha=0.85, personalization=personalization, weight="weight"
        )
        graph_state["pagerank"] = pr

        # Update underratedness scores and persist to database
        for node_id, score in pr.items():
            if node_id in graph_state["nodes"]:
                node = graph_state["nodes"][node_id]
                followers = node.get("followers_count", 1)
                node["pagerank_score"] = score
                node["underratedness_score"] = score / math.log(1 + max(followers, 1))

                # Persist PageRank score to database
                try:
                    update_node_pagerank(node_id, score)
                except Exception:
                    pass  # Non-critical, continue
    except Exception as e:
        print(f"PageRank error: {e}")


def add_user_to_graph(user: Dict[str, Any], is_seed: bool = False, depth: int = 0):
    """Add a user node to the graph and persist to database."""
    uid = user.get("id", "")
    if not uid:
        return None

    G = graph_state["graph"]
    G.add_node(uid)

    metrics = user.get("public_metrics", {})

    node_data = {
        "id": uid,
        "handle": user.get("username", f"id_{uid}"),
        "name": user.get("name", ""),
        "bio": user.get("description", "")[:500],
        "followers_count": metrics.get("followers_count", 0),
        "following_count": metrics.get("following_count", 0),
        "tweet_count": metrics.get("tweet_count", 0),
        "is_seed": is_seed,
        "is_candidate": not is_seed
        and 50 <= metrics.get("followers_count", 0) <= 50000,
        "pagerank_score": 0.0,
        "underratedness_score": 0.0,
        "depth": depth,
    }

    # Merge with existing data if present
    if uid in graph_state["nodes"]:
        existing = graph_state["nodes"][uid]
        existing.update(node_data)
        existing["is_seed"] = existing.get("is_seed", False) or is_seed
        node_data = existing
    else:
        graph_state["nodes"][uid] = node_data

    if is_seed:
        graph_state["seeds"].add(uid)

    # Persist to database
    try:
        upsert_graph_node(node_data)
    except Exception as e:
        print(f"[graph] Error persisting node {uid}: {e}")

    return node_data


def add_edge_to_graph(
    src_id: str, dst_id: str, interaction_type: str = "follow", weight: float = 5.0
):
    """Add an edge to the graph and persist to database."""
    G = graph_state["graph"]
    G.add_edge(src_id, dst_id, weight=weight, interaction_type=interaction_type)

    # Persist to database
    try:
        add_graph_edge(src_id, dst_id, weight=weight, interaction_type=interaction_type)
    except Exception as e:
        print(f"[graph] Error persisting edge {src_id} -> {dst_id}: {e}")


async def expand_from_source(handle: str, depth: int, max_following: int):
    """Expand the graph from a source account."""
    try:
        graph_state["loading"] = True
        graph_state["last_error"] = None

        # Clean up handle
        handle = handle.strip().lstrip("@").strip()
        print(f"[expand] Looking up @{handle}")

        client = get_x_client()

        # Get the source user
        try:
            user = client.get_user_by_username(handle)
        except Exception as e:
            graph_state["last_error"] = f"X API error for @{handle}: {str(e)}"
            graph_state["loading"] = False
            return

        if not user:
            graph_state["last_error"] = f"Could not find user @{handle}"
            graph_state["loading"] = False
            return

        print(f"[expand] Found user: @{user.get('username')} (ID: {user.get('id')})")

        # Add as seed node
        add_user_to_graph(user, is_seed=True, depth=0)
        root_id = user["id"]

        # Expand following at each depth level
        current_frontier = [root_id]

        for current_depth in range(1, depth + 1):
            next_frontier = []

            for user_id in current_frontier:
                following = client.get_user_following(
                    user_id, max_results=max_following
                )

                for fol_user in following:
                    fol_id = fol_user.get("id")
                    if not fol_id:
                        continue

                    add_user_to_graph(fol_user, is_seed=False, depth=current_depth)
                    add_edge_to_graph(user_id, fol_id, "follow", 5.0)

                    if current_depth < depth and fol_id not in next_frontier:
                        next_frontier.append(fol_id)

            current_frontier = next_frontier[:20]  # Limit frontier expansion

        # Recompute PageRank
        compute_pagerank_scores()

        graph_state["loading"] = False

    except Exception as e:
        graph_state["last_error"] = str(e)
        graph_state["loading"] = False
        raise


def load_graph_from_pickle():
    """Load graph from database into in-memory state."""
    try:
        print("[startup] Loading graph from database...")

        # Load nodes from database
        db_nodes = get_all_graph_nodes()
        db_edges = get_all_graph_edges()

        if not db_nodes:
            print("[startup] No existing graph data in database")
            return False

        graph_state["graph"] = nx.DiGraph()
        graph_state["seeds"] = set()
        graph_state["nodes"] = {}

        # Add nodes to graph and state
        for node in db_nodes:
            node_id = node["id"]
            graph_state["nodes"][node_id] = node
            graph_state["graph"].add_node(node_id, **node)

            if node.get("is_seed"):
                graph_state["seeds"].add(node_id)

        # Add edges to graph
        for edge in db_edges:
            graph_state["graph"].add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 1.0),
                interaction_type=edge.get("interaction_type", "follow"),
            )

        print(
            f"[startup] Loaded {len(graph_state['nodes'])} nodes, "
            f"{graph_state['graph'].number_of_edges()} edges, "
            f"{len(graph_state['seeds'])} seeds from database"
        )
        return True
    except Exception as e:
        print(f"[startup] Error loading graph from database: {e}")
        return False


def save_graph_to_pickle():
    """Save current graph state to pickle file."""
    try:
        G = graph_state["graph"]

        # Update node attributes in the graph
        for node_id, node_data in graph_state["nodes"].items():
            if node_id in G:
                G.nodes[node_id].update(
                    {
                        "handle": node_data.get("handle", ""),
                        "name": node_data.get("name", ""),
                        "bio": node_data.get("bio", ""),
                        "followers_count": node_data.get("followers_count", 0),
                        "following_count": node_data.get("following_count", 0),
                        "is_root": node_data.get("is_seed", False),
                        "is_candidate": node_data.get("is_candidate", False),
                    }
                )

        GRAPH_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PICKLE_PATH, "wb") as f:
            pickle.dump(G, f)
        print(f"[save] Saved graph to {GRAPH_PICKLE_PATH}")
        return True
    except Exception as e:
        print(f"[save] Error saving graph: {e}")
        return False


def load_deep_evaluations() -> List[dict]:
    """Load all deep evaluation results from files, fallback to database."""
    evaluations = []
    seen_handles = set()

    # Try loading from files first (for local development)
    if ENRICHED_DIR.exists():
        # Load individual deep_*.json files
        for path in ENRICHED_DIR.glob("deep_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    handle = data.get("handle", "").lower()
                    if handle and handle not in seen_handles:
                        evaluations.append(data)
                        seen_handles.add(handle)
            except (json.JSONDecodeError, KeyError):
                continue

        # Also load from candidates_deep_eval.json if it exists
        bulk_eval_path = ENRICHED_DIR / "candidates_deep_eval.json"
        if bulk_eval_path.exists():
            try:
                with open(bulk_eval_path) as f:
                    bulk_data = json.load(f)
                    if isinstance(bulk_data, list):
                        for data in bulk_data:
                            handle = data.get("handle", "").lower()
                            if handle and handle not in seen_handles:
                                evaluations.append(data)
                                seen_handles.add(handle)
            except (json.JSONDecodeError, KeyError):
                pass

    # If no file evaluations found, fallback to database
    if not evaluations:
        try:
            db_candidates = get_evaluated_candidates()
            for c in db_candidates:
                # Convert database format to evaluation format
                # get_evaluated_candidates() now returns deep_eval_data fields
                evaluations.append({
                    "handle": c.get("handle", ""),
                    "name": c.get("name", ""),
                    "bio": c.get("bio", ""),
                    "followers": c.get("followers_count", 0),
                    "final_score": c.get("final_score", 0),
                    "recommended_role": c.get("recommended_role") or c.get("grok_role") or "none",
                    "summary": c.get("summary", ""),
                    "strengths": c.get("strengths", []),
                    "concerns": c.get("concerns", []),
                    "pagerank_score": c.get("pagerank_score", 0),
                    "underratedness_score": c.get("underratedness_score", 0),
                    # Include detailed score breakdowns from deep eval
                    "technical_depth": c.get("technical_depth"),
                    "project_evidence": c.get("project_evidence"),
                    "mission_alignment": c.get("mission_alignment"),
                    "exceptional_ability": c.get("exceptional_ability"),
                    "communication": c.get("communication"),
                    "github_url": c.get("github_url"),
                    "linkedin_url": c.get("linkedin_url"),
                    "top_repos": c.get("top_repos"),
                })
        except Exception as e:
            print(f"[eval] Error loading from database: {e}")

    return evaluations


def load_graph_nodes() -> dict:
    """Load node data from processed CSV, fallback to database."""
    import csv

    nodes = {}
    nodes_path = PROCESSED_DIR / "nodes.csv"

    # Try loading from CSV file first (for local development)
    if nodes_path.exists():
        with open(nodes_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                handle = row.get("handle", "")
                if handle:
                    nodes[handle.lower()] = row

    # If no nodes from file, fallback to database
    if not nodes:
        try:
            db_nodes = get_all_graph_nodes(limit=50000)
            for node in db_nodes:
                handle = node.get("handle", "")
                if handle:
                    nodes[handle.lower()] = node
        except Exception as e:
            print(f"[nodes] Error loading from database: {e}")

    return nodes


# Cache for loaded data
_evaluation_cache = None
_node_cache = None


def get_evaluations():
    """Get cached evaluations."""
    global _evaluation_cache
    if _evaluation_cache is None:
        _evaluation_cache = load_deep_evaluations()
    return _evaluation_cache


def get_nodes():
    """Get cached nodes."""
    global _node_cache
    if _node_cache is None:
        _node_cache = load_graph_nodes()
    return _node_cache


def evaluation_to_list_item(
    eval_data: dict, node_data: dict = None
) -> CandidateListItem:
    """Convert evaluation dict to CandidateListItem with score breakdown."""
    # Build score breakdowns if available
    technical_depth = None
    if eval_data.get("technical_depth"):
        technical_depth = ScoreBreakdown(
            score=eval_data["technical_depth"].get("score", 0),
            evidence=eval_data["technical_depth"].get("evidence", ""),
        )

    project_evidence = None
    if eval_data.get("project_evidence"):
        project_evidence = ScoreBreakdown(
            score=eval_data["project_evidence"].get("score", 0),
            evidence=eval_data["project_evidence"].get("evidence", ""),
        )

    mission_alignment = None
    if eval_data.get("mission_alignment"):
        mission_alignment = ScoreBreakdown(
            score=eval_data["mission_alignment"].get("score", 0),
            evidence=eval_data["mission_alignment"].get("evidence", ""),
        )

    exceptional_ability = None
    if eval_data.get("exceptional_ability"):
        exceptional_ability = ScoreBreakdown(
            score=eval_data["exceptional_ability"].get("score", 0),
            evidence=eval_data["exceptional_ability"].get("evidence", ""),
        )

    communication = None
    if eval_data.get("communication"):
        communication = ScoreBreakdown(
            score=eval_data["communication"].get("score", 0),
            evidence=eval_data["communication"].get("evidence", ""),
        )

    return CandidateListItem(
        handle=eval_data.get("handle", ""),
        name=node_data.get("name", "") if node_data else "",
        bio=eval_data.get("bio", "")[:200],
        followers_count=eval_data.get("followers", 0),
        final_score=eval_data.get("final_score", 0.0),
        recommended_role=eval_data.get("recommended_role", "none"),
        summary=eval_data.get("summary", ""),
        strengths=eval_data.get("strengths", [])[:3],
        pagerank_score=float(node_data.get("pagerank_score", 0)) if node_data else 0.0,
        underratedness_score=float(node_data.get("underratedness_score", 0))
        if node_data
        else 0.0,
        github_url=eval_data.get("github_url"),
        linkedin_url=eval_data.get("linkedin_url"),
        technical_depth=technical_depth,
        project_evidence=project_evidence,
        mission_alignment=mission_alignment,
        exceptional_ability=exceptional_ability,
        communication=communication,
    )


def evaluation_to_detail(eval_data: dict, node_data: dict = None) -> CandidateDeepEval:
    """Convert evaluation dict to CandidateDeepEval."""
    # Helper to safely get score breakdown (handles None values)
    def get_breakdown(key: str) -> ScoreBreakdown:
        data = eval_data.get(key)
        if data and isinstance(data, dict):
            return ScoreBreakdown(
                score=data.get("score", 0),
                evidence=data.get("evidence", ""),
            )
        return ScoreBreakdown(score=0, evidence="")

    return CandidateDeepEval(
        handle=eval_data.get("handle", ""),
        name=node_data.get("name", "") if node_data else "",
        bio=eval_data.get("bio", ""),
        followers_count=eval_data.get("followers", 0),
        final_score=eval_data.get("final_score", 0.0),
        technical_depth=get_breakdown("technical_depth"),
        project_evidence=get_breakdown("project_evidence"),
        mission_alignment=get_breakdown("mission_alignment"),
        exceptional_ability=get_breakdown("exceptional_ability"),
        communication=get_breakdown("communication"),
        summary=eval_data.get("summary", ""),
        strengths=eval_data.get("strengths", []),
        concerns=eval_data.get("concerns", []),
        recommended_role=eval_data.get("recommended_role", "none"),
        github_url=eval_data.get("github_url"),
        linkedin_url=eval_data.get("linkedin_url"),
        top_repos=eval_data.get("top_repos"),
        pagerank_score=float(node_data.get("pagerank_score", 0)) if node_data else 0.0,
        underratedness_score=float(node_data.get("underratedness_score", 0))
        if node_data
        else 0.0,
        discovered_via=node_data.get("discovered_via", "") if node_data else "",
    )


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "Grok Recruiter API",
        "version": "1.0.0",
        "endpoints": [
            "/candidates",
            "/candidates/{handle}",
            "/search",
            "/stats",
        ],
    }


@app.get("/candidates", response_model=List[CandidateListItem])
async def list_candidates(
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query(
        "final_score", enum=["final_score", "underratedness_score", "pagerank_score"]
    ),
    role: Optional[str] = Query(
        None, enum=["research", "engineering", "infrastructure"]
    ),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
):
    """
    List candidates with filtering and sorting.

    - **limit**: Maximum candidates to return (default 30)
    - **offset**: Pagination offset
    - **sort_by**: Sort field (final_score, underratedness_score, pagerank_score)
    - **role**: Filter by recommended role
    - **min_score**: Minimum final_score filter
    """
    evaluations = get_evaluations()
    nodes = get_nodes()

    # Filter
    filtered = []
    for e in evaluations:
        if e.get("final_score", 0) < min_score:
            continue
        if role and e.get("recommended_role") != role:
            continue
        filtered.append(e)

    # Sort
    if sort_by == "final_score":
        filtered.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    elif sort_by == "underratedness_score":
        # Need to merge with node data
        for e in filtered:
            handle = e.get("handle", "").lower()
            node = nodes.get(handle, {})
            e["_sort"] = float(node.get("underratedness_score", 0))
        filtered.sort(key=lambda x: x.get("_sort", 0), reverse=True)
    elif sort_by == "pagerank_score":
        for e in filtered:
            handle = e.get("handle", "").lower()
            node = nodes.get(handle, {})
            e["_sort"] = float(node.get("pagerank_score", 0))
        filtered.sort(key=lambda x: x.get("_sort", 0), reverse=True)

    # Paginate
    paginated = filtered[offset : offset + limit]

    # Convert to response
    results = []
    for e in paginated:
        handle = e.get("handle", "").lower()
        node = nodes.get(handle, {})
        results.append(evaluation_to_list_item(e, node))

    return results


@app.get("/candidates/{handle}", response_model=CandidateDetailResponse)
async def get_candidate(handle: str):
    """Get detailed candidate information."""
    evaluations = get_evaluations()
    nodes = get_nodes()

    # Find evaluation by handle
    handle_lower = handle.lower().lstrip("@")
    eval_data = None
    for e in evaluations:
        if e.get("handle", "").lower() == handle_lower:
            eval_data = e
            break

    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Candidate @{handle} not found")

    node_data = nodes.get(handle_lower, {})

    return CandidateDetailResponse(
        candidate=evaluation_to_detail(eval_data, node_data),
    )


def stream_grok_search(query: str, candidates_summary: str):
    """Stream Grok search with real-time reasoning output using grok-3-mini."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "API key not configured"}
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = f"""You are helping find the best candidates for a recruiting search.

USER QUERY: "{query}"

AVAILABLE CANDIDATES (handle: summary):
{candidates_summary}

Analyze the query and identify ONLY the candidates who are genuinely relevant.
Do NOT include candidates who don't match the query criteria.
Be selective - it's better to return fewer highly relevant candidates than many loosely related ones.

Respond with JSON only:
{{
  "criteria": "Brief description of what you're looking for based on the query",
  "rankings": ["handle1", "handle2", ...]  // ONLY include relevant candidates, ordered by relevance
}}"""

    try:
        # Use grok-3-mini with streaming
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json={
                "model": "grok-3-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert technical recruiter.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "reasoning_effort": "low",
                "stream": True,
            },
            timeout=120,
            stream=True,
        )

        if response.status_code != 200:
            yield {"type": "error", "message": f"API error: {response.status_code}"}
            return

        reasoning_content = ""
        content = ""

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8")
            if not line_text.startswith("data: "):
                continue

            data_str = line_text[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Stream reasoning content (thinking)
                if "reasoning_content" in delta:
                    reasoning_chunk = delta["reasoning_content"]
                    reasoning_content += reasoning_chunk
                    yield {"type": "thinking", "content": reasoning_chunk}

                # Stream final content
                if "content" in delta:
                    content_chunk = delta["content"]
                    content += content_chunk

            except json.JSONDecodeError:
                continue

        # Parse final JSON response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        try:
            parsed = json.loads(content)
            yield {
                "type": "result",
                "criteria": parsed.get("criteria", ""),
                "rankings": parsed.get("rankings", []),
            }
        except json.JSONDecodeError:
            yield {"type": "result", "criteria": "", "rankings": []}

    except Exception as e:
        print(f"Grok stream error: {e}")
        yield {"type": "error", "message": str(e)}


@app.post("/search/stream")
async def search_candidates_stream(request: SearchRequest):
    """
    Streaming search endpoint with real-time Grok reasoning.
    Returns Server-Sent Events (SSE).
    """
    evaluations = get_evaluations()
    nodes = get_nodes()

    # Filter candidates first
    filtered = []
    for e in evaluations:
        if request.min_score > 0 and e.get("final_score", 0) < request.min_score:
            continue
        if request.role_filter and e.get("recommended_role") != request.role_filter:
            continue
        filtered.append(e)

    # Build summary for Grok
    candidates_summary = "\n".join(
        f"@{e.get('handle', '')}: {e.get('summary', '')[:150]}" for e in filtered[:50]
    )

    handle_to_eval = {e.get("handle", "").lower(): e for e in filtered}

    def generate():
        rankings = []
        criteria = ""

        # Stream Grok response
        for chunk in stream_grok_search(request.query, candidates_summary):
            if chunk["type"] == "thinking":
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "result":
                rankings = chunk.get("rankings", [])
                criteria = chunk.get("criteria", "")
            elif chunk["type"] == "error":
                yield f"data: {json.dumps(chunk)}\n\n"

        # Build final candidates list - ONLY include Grok-ranked candidates
        ranked = []
        seen = set()

        for handle in rankings:
            handle_lower = handle.lower().lstrip("@")
            if handle_lower in handle_to_eval and handle_lower not in seen:
                ranked.append(handle_to_eval[handle_lower])
                seen.add(handle_lower)

        # Convert to response format (only relevant candidates)
        results = []
        for e in ranked[: request.limit]:
            handle = e.get("handle", "").lower()
            node = nodes.get(handle, {})
            item = evaluation_to_list_item(e, node)
            results.append(item.model_dump())

        # Send final results
        yield f"data: {json.dumps({'type': 'candidates', 'candidates': results, 'criteria': criteria, 'total': len(ranked)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest):
    """
    Non-streaming search endpoint (fallback) - filters to only matching candidates.
    """
    evaluations = get_evaluations()
    nodes = get_nodes()

    filtered = []
    for e in evaluations:
        if request.min_score > 0 and e.get("final_score", 0) < request.min_score:
            continue
        if request.role_filter and e.get("recommended_role") != request.role_filter:
            continue
        filtered.append(e)

    # Simple keyword filtering - only include candidates with keyword matches
    query = request.query.lower()
    keywords = [k.strip() for k in query.split() if len(k) > 2]

    matched = []
    for e in filtered:
        text = " ".join(
            [
                e.get("bio", ""),
                e.get("summary", ""),
                " ".join(e.get("strengths", [])),
                e.get("technical_depth", {}).get("evidence", ""),
            ]
        ).lower()
        match_score = sum(1 for kw in keywords if kw in text)
        # Only include if at least one keyword matches
        if match_score > 0:
            matched.append((e, match_score + e.get("final_score", 0) / 100))

    matched.sort(key=lambda x: x[1], reverse=True)

    results = []
    for e, _ in matched[: request.limit]:
        handle = e.get("handle", "").lower()
        node = nodes.get(handle, {})
        results.append(evaluation_to_list_item(e, node))

    return SearchResponse(
        query=request.query,
        total_results=len(matched),
        candidates=results,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics from database with file/memory fallbacks."""
    # Get stats from database
    db_stats = get_graph_stats()

    # Count file-based evaluations as fallback
    fast_screen_dir = DATA_DIR / "evaluations" / "fast_screen"
    fast_screened_files = (
        len(list(fast_screen_dir.glob("*.json"))) if fast_screen_dir.exists() else 0
    )

    # Count deep evals from enriched directory
    enriched_dir = DATA_DIR / "enriched"
    deep_eval_files = (
        len([f for f in enriched_dir.glob("deep_*.json")]) if enriched_dir.exists() else 0
    )

    # Fallback to in-memory graph if database is empty
    total_nodes = db_stats.get("total_nodes", 0)
    seed_count = db_stats.get("seeds", 0)
    if total_nodes == 0 and graph_state["nodes"]:
        total_nodes = len(graph_state["nodes"])
        seed_count = sum(1 for n in graph_state["nodes"].values() if n.get("is_seed"))

    return StatsResponse(
        total_nodes=total_nodes,
        total_candidates=db_stats.get("candidates", 0) or len([n for n in graph_state["nodes"].values() if n.get("is_candidate")]),
        fast_screened=db_stats.get("filtered_count", 0) or fast_screened_files,
        deep_evaluated=db_stats.get("evaluated_count", 0) or deep_eval_files,
        seed_accounts=seed_count,
    )


@app.post("/reload")
async def reload_data():
    """Force reload data from disk."""
    global _evaluation_cache, _node_cache
    _evaluation_cache = None
    _node_cache = None
    return {
        "status": "ok",
        "message": "Cache cleared, data will reload on next request",
    }


# --- Saved Candidates Endpoints ---


@app.post("/saved", response_model=SavedCandidateResponse)
async def save_candidate(request: SaveCandidateRequest, user: dict = Depends(require_auth)):
    """Save a candidate to the database for the authenticated user."""
    handle = request.handle.lower().lstrip("@")
    user_id = user.get("handle", "").lower()

    # Verify candidate exists
    evaluations = get_evaluations()
    exists = any(e.get("handle", "").lower() == handle for e in evaluations)
    if not exists:
        raise HTTPException(status_code=404, detail=f"Candidate @{handle} not found")

    try:
        result = db_save_candidate(handle, user_id, request.notes)
        return SavedCandidateResponse(**result)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Candidate already saved")


@app.delete("/saved/{handle}")
async def unsave_candidate(handle: str, user: dict = Depends(require_auth)):
    """Remove a candidate from the user's saved list."""
    handle = handle.lower().lstrip("@")
    user_id = user.get("handle", "").lower()

    if not db_unsave_candidate(handle, user_id):
        raise HTTPException(status_code=404, detail="Candidate not in saved list")

    return {"status": "ok", "handle": handle}


@app.get("/saved", response_model=SavedCandidatesListResponse)
async def list_saved_candidates(user: dict = Depends(require_auth)):
    """Get all saved candidates with their full evaluation data for the authenticated user."""
    evaluations = get_evaluations()
    nodes = get_nodes()
    user_id = user.get("handle", "").lower()

    saved_handles = set(db_get_saved_handles(user_id))

    # Get full candidate data for saved handles
    results = []
    for e in evaluations:
        handle = e.get("handle", "").lower()
        if handle in saved_handles:
            node = nodes.get(handle, {})
            results.append(evaluation_to_list_item(e, node))

    return SavedCandidatesListResponse(saved=results, total=len(results))


@app.get("/saved/handles")
async def get_saved_handles(user: dict = Depends(require_auth)):
    """Get list of all saved candidate handles for the authenticated user."""
    user_id = user.get("handle", "").lower()
    handles = db_get_saved_handles(user_id)
    return {"handles": handles}


@app.get("/saved/count")
async def saved_count(user: dict = Depends(require_auth)):
    """Get count of saved candidates for the authenticated user."""
    user_id = user.get("handle", "").lower()
    count = get_saved_count(user_id)
    return {"count": count}


@app.get("/saved/check/{handle}")
async def check_if_saved(handle: str, user: dict = Depends(require_auth)):
    """Check if a specific candidate is saved by the authenticated user."""
    handle = handle.lower().lstrip("@")
    user_id = user.get("handle", "").lower()
    is_saved = is_candidate_saved(handle, user_id)
    return {"is_saved": is_saved}


# --- DM Generation Endpoints ---

DM_SYSTEM_PROMPT = """You are an expert technical recruiter writing personalized outreach messages for exceptional candidates. Your messages should:

1. Be authentic and respectful of the candidate's time
2. Reference SPECIFIC work, projects, or tweets that caught your attention
3. Clearly explain why they're a fit for the role/team
4. Keep it concise (under 280 characters for X DM, or 2-3 short paragraphs)
5. Include a clear, low-pressure call-to-action
6. Avoid corporate jargon and generic flattery

Never mention that you're an AI or that this message was auto-generated."""


def build_dm_prompt(eval_data: dict, custom_context: Optional[str], tone: str) -> str:
    """Build the prompt for DM generation."""
    tone_instructions = {
        "professional": "Use a professional but warm tone. Be direct and respectful.",
        "casual": "Use a casual, friendly tone. Feel free to use contractions and be conversational.",
        "enthusiastic": "Use an enthusiastic, energetic tone. Show genuine excitement about their work.",
    }

    # Build score details if available
    score_details = []
    for criterion in [
        "technical_depth",
        "project_evidence",
        "mission_alignment",
        "exceptional_ability",
        "communication",
    ]:
        if criterion in eval_data and eval_data[criterion]:
            data = eval_data[criterion]
            if isinstance(data, dict):
                score_details.append(
                    f"  - {criterion.replace('_', ' ').title()}: {data.get('score', 'N/A')}/10 - {data.get('evidence', '')}"
                )

    score_section = "\n".join(score_details) if score_details else "Not available"

    # Optional recruiter context section
    context_section = ""
    if custom_context and custom_context.strip():
        context_section = f"""
ADDITIONAL CONTEXT FROM RECRUITER:
{custom_context.strip()}
"""

    return f"""Generate a personalized outreach message for this candidate:

CANDIDATE PROFILE:
- Handle: @{eval_data.get("handle", "")}
- Name: {eval_data.get("name", "")}
- Bio: {eval_data.get("bio", "")}
- Followers: {eval_data.get("followers_count", 0):,}
- Recommended Role: {eval_data.get("recommended_role", "")}
- GitHub: {eval_data.get("github_url", "Not available")}
- LinkedIn: {eval_data.get("linkedin_url", "Not available")}

EVALUATION SUMMARY:
{eval_data.get("summary", "")}

KEY STRENGTHS:
{chr(10).join("- " + s for s in eval_data.get("strengths", [])[:4])}

DETAILED SCORES:
{score_section}
{context_section}
TONE: {tone_instructions.get(tone, tone_instructions["professional"])}

Write a compelling outreach message that:
1. Opens with something SPECIFIC about their work - reference their actual strengths, projects, or expertise
2. Shows you've done your research and genuinely appreciate their contributions
3. Mentions an opportunity that fits their skills (e.g., AI/ML roles at a cutting-edge company)
4. Ends with a clear but low-pressure next step (e.g., "Would love to chat if you're open to it")

Keep the message under 280 characters if possible (for X DM). Output ONLY the message text, no quotes or labels."""


@app.post("/dm/generate/stream")
async def generate_dm_stream(request: DMGenerateRequest):
    """Stream personalized DM generation for real-time preview."""
    handle = request.handle.lower().lstrip("@")

    # Get candidate data
    evaluations = get_evaluations()
    eval_data = None
    for e in evaluations:
        if e.get("handle", "").lower() == handle:
            eval_data = e
            break

    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Candidate @{handle} not found")

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    def generate():
        prompt = build_dm_prompt(eval_data, request.custom_context, request.tone)
        full_message = ""

        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-3-mini",
                    "messages": [
                        {"role": "system", "content": DM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "stream": True,
                },
                timeout=120,
                stream=True,
            )

            if response.status_code != 200:
                yield f"data: {json.dumps({'type': 'error', 'message': f'API error: {response.status_code}'})}\n\n"
                return

            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue

                data_str = line_text[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if content:
                        full_message += content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                except json.JSONDecodeError:
                    continue

            # Save to history
            save_dm_history(handle, request.custom_context, full_message)

            # Send final result with X intent URL
            x_dm_url = (
                f"https://twitter.com/messages/compose?text={quote(full_message)}"
            )
            yield f"data: {json.dumps({'type': 'done', 'message': full_message, 'x_intent_url': x_dm_url, 'character_count': len(full_message)})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/dm/generate", response_model=DMGenerateResponse)
async def generate_dm(request: DMGenerateRequest):
    """Non-streaming DM generation."""
    handle = request.handle.lower().lstrip("@")

    # Get candidate data
    evaluations = get_evaluations()
    eval_data = None
    for e in evaluations:
        if e.get("handle", "").lower() == handle:
            eval_data = e
            break

    if not eval_data:
        raise HTTPException(status_code=404, detail=f"Candidate @{handle} not found")

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    prompt = build_dm_prompt(eval_data, request.custom_context, request.tone)

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3-mini",
                "messages": [
                    {"role": "system", "content": DM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="DM generation failed")

        message = response.json()["choices"][0]["message"]["content"].strip()

        # Save to history
        save_dm_history(handle, request.custom_context, message)

        # Create X intent URL
        x_dm_url = f"https://twitter.com/messages/compose?text={quote(message)}"

        return DMGenerateResponse(
            handle=handle,
            message=message,
            x_intent_url=x_dm_url,
            character_count=len(message),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Graph Visualization Endpoints ---


@app.get("/status", response_model=GraphStatusResponse)
async def get_graph_status():
    """Get current graph loading status."""
    return GraphStatusResponse(
        loading=graph_state["loading"],
        node_count=len(graph_state["nodes"]),
        edge_count=graph_state["graph"].number_of_edges(),
        seed_count=len(graph_state["seeds"]),
        last_error=graph_state["last_error"],
    )


@app.get("/graph", response_model=GraphResponse)
async def get_graph(
    min_pagerank: float = 0.0,
    max_nodes: int = 5000,
    only_relevant: bool = False,
    max_depth: int = 10,
):
    """Get the current graph data for visualization."""
    nodes_data = []
    edges_data = []

    # Get nodes sorted by PageRank
    sorted_nodes = sorted(
        graph_state["nodes"].values(),
        key=lambda n: n.get("pagerank_score", 0),
        reverse=True,
    )

    # Filter and limit
    node_ids = set()
    for node in sorted_nodes:
        if len(node_ids) >= max_nodes:
            break

        pr_score = node.get("pagerank_score", 0)
        if pr_score < min_pagerank and not node.get("is_seed"):
            continue

        # Filter by depth
        node_depth = node.get("depth", 0)
        if node_depth > max_depth and not node.get("is_seed"):
            continue

        # Filter by Grok relevance if requested
        if only_relevant:
            filter_result = graph_state["filter_results"].get(node["id"])
            if filter_result and not filter_result.pass_filter:
                continue

        node_ids.add(node["id"])

        # Get Grok filter info if available
        filter_result = graph_state["filter_results"].get(node["id"])

        nodes_data.append(
            GraphNode(
                id=node["id"],
                handle=node.get("handle", ""),
                name=node.get("name", ""),
                bio=node.get("bio", ""),
                followers_count=node.get("followers_count", 0),
                following_count=node.get("following_count", 0),
                is_seed=node.get("is_seed", False),
                is_candidate=node.get("is_candidate", False),
                pagerank_score=node.get("pagerank_score", 0),
                underratedness_score=node.get("underratedness_score", 0),
                grok_relevant=filter_result.pass_filter if filter_result else None,
                grok_role=filter_result.potential_role if filter_result else None,
                depth=node.get("depth", 0),
                discovered_via=node.get("discovered_via"),
                submission_pending=node.get("submission_pending"),
            )
        )

    # Get edges between visible nodes
    G = graph_state["graph"]
    for src, dst, data in G.edges(data=True):
        if src in node_ids and dst in node_ids:
            edges_data.append(
                GraphEdge(
                    source=src,
                    target=dst,
                    weight=data.get("weight", 1.0),
                    interaction_type=data.get("interaction_type", "follow"),
                )
            )

    # Compute stats
    stats = {
        "total_nodes": len(graph_state["nodes"]),
        "total_edges": G.number_of_edges(),
        "displayed_nodes": len(nodes_data),
        "displayed_edges": len(edges_data),
        "seeds": len(graph_state["seeds"]),
        "filtered_count": sum(
            1 for r in graph_state["filter_results"].values() if r.pass_filter
        ),
    }

    return GraphResponse(nodes=nodes_data, edges=edges_data, stats=stats)


@app.post("/source/add")
async def add_source(request: AddSourceRequest, background_tasks: BackgroundTasks):
    """Add a new source account and expand the graph."""
    if graph_state["loading"]:
        raise HTTPException(status_code=409, detail="Graph is currently loading")

    # Run expansion in background
    background_tasks.add_task(
        expand_from_source,
        request.handle,
        request.depth,
        request.max_following,
    )

    return {
        "message": f"Started expanding from @{request.handle}",
        "depth": request.depth,
    }


@app.post("/filter")
async def filter_nodes(request: FilterRequest, background_tasks: BackgroundTasks):
    """Filter nodes using Grok fast screening."""
    if graph_state["loading"]:
        raise HTTPException(status_code=409, detail="Graph is currently loading")

    async def run_grok_filter(query: str):
        try:
            graph_state["loading"] = True
            client = get_grok_client()

            # Filter candidates
            candidates = [
                n
                for n in graph_state["nodes"].values()
                if n.get("is_candidate")
                and n["id"] not in graph_state["filter_results"]
            ]

            for i, node in enumerate(candidates[:200]):  # Limit to 200 for speed
                result = client.fast_screen(
                    handle=node["handle"],
                    bio=node.get("bio", ""),
                    pinned_tweet=None,
                    location=None,
                )
                graph_state["filter_results"][node["id"]] = result

                if (i + 1) % 20 == 0:
                    print(f"Filtered {i + 1}/{len(candidates[:200])} nodes")

            graph_state["loading"] = False
        except Exception as e:
            graph_state["last_error"] = str(e)
            graph_state["loading"] = False

    background_tasks.add_task(run_grok_filter, request.query)

    return {"message": "Started Grok filtering"}


@app.delete("/graph")
async def clear_graph():
    """Clear the entire graph."""
    graph_state["graph"] = nx.DiGraph()
    graph_state["nodes"] = {}
    graph_state["pagerank"] = {}
    graph_state["filter_results"] = {}
    graph_state["seeds"] = set()
    graph_state["loading"] = False
    graph_state["last_error"] = None

    return {"message": "Graph cleared"}


@app.post("/load")
async def load_existing_graph():
    """Load graph from existing pickle file."""
    if load_graph_from_pickle():
        return {
            "message": "Graph loaded",
            "nodes": len(graph_state["nodes"]),
            "edges": graph_state["graph"].number_of_edges(),
            "seeds": len(graph_state["seeds"]),
        }
    raise HTTPException(status_code=404, detail="No existing graph found")


@app.post("/save")
async def save_graph():
    """Save current graph to pickle file."""
    if save_graph_to_pickle():
        return {"message": "Graph saved", "path": str(GRAPH_PICKLE_PATH)}
    raise HTTPException(status_code=500, detail="Failed to save graph")


@app.post("/graph/import")
async def import_graph(request: Request):
    """Bulk import nodes and edges into the graph database."""
    try:
        data = await request.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        nodes_added = 0
        edges_added = 0

        # Import nodes
        for node in nodes:
            # Handle is_seed - can come from is_seed or is_root
            is_seed = node.get("is_seed")
            if is_seed is None:
                is_seed = str(node.get("is_root", "")).lower() == "true" or node.get("is_root") is True
            elif isinstance(is_seed, str):
                is_seed = is_seed.lower() == "true"

            node_data = {
                "id": str(node.get("user_id", node.get("id", ""))),
                "handle": node.get("handle", ""),
                "name": node.get("name", ""),
                "bio": node.get("bio", "")[:500] if node.get("bio") else "",
                "followers_count": int(node.get("followers_count", 0)),
                "following_count": int(node.get("following_count", 0)),
                "is_seed": is_seed,
                "is_candidate": str(node.get("is_candidate", "")).lower() == "true" or node.get("is_candidate") is True,
                "discovered_via": node.get("discovered_via", ""),
                "depth": int(node.get("depth", 0)) if node.get("depth") else 0,
                "pagerank_score": float(node.get("pagerank_score", 0)) if node.get("pagerank_score") else 0.0,
                "underratedness_score": float(node.get("underratedness_score", 0)) if node.get("underratedness_score") else 0.0,
                # Screening fields
                "grok_relevant": node.get("grok_relevant"),
                "grok_role": node.get("grok_role"),
                "grok_evaluated": node.get("grok_evaluated", False),
                "final_score": node.get("final_score"),
                # Full deep eval data as JSON string
                "deep_eval_data": node.get("deep_eval_data"),
            }

            if node_data["id"]:
                # Add to in-memory state
                graph_state["nodes"][node_data["id"]] = node_data
                graph_state["graph"].add_node(node_data["id"], **node_data)
                if node_data["is_seed"]:
                    graph_state["seeds"].add(node_data["id"])

                # Persist to database
                try:
                    upsert_graph_node(node_data)
                    nodes_added += 1
                except Exception as e:
                    print(f"[import] Error persisting node {node_data['id']}: {e}")

        # Import edges
        for edge in edges:
            src = str(edge.get("src_user_id", edge.get("source", "")))
            dst = str(edge.get("dst_user_id", edge.get("target", "")))
            weight = float(edge.get("weight", 1.0)) if edge.get("weight") else 1.0
            interaction_type = edge.get("interaction_type", "follow")

            if src and dst:
                # Add to in-memory state
                graph_state["graph"].add_edge(src, dst, weight=weight, interaction_type=interaction_type)

                # Persist to database
                try:
                    add_graph_edge(src, dst, weight=weight, interaction_type=interaction_type)
                    edges_added += 1
                except Exception as e:
                    print(f"[import] Error persisting edge {src} -> {dst}: {e}")

        return {
            "message": "Import completed",
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "total_nodes": len(graph_state["nodes"]),
            "total_edges": graph_state["graph"].number_of_edges(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@app.get("/node/{node_id}")
async def get_node_details(node_id: str):
    """Get detailed information about a specific node."""
    if node_id not in graph_state["nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")

    node = graph_state["nodes"][node_id]
    filter_result = graph_state["filter_results"].get(node_id)

    # Get connections
    G = graph_state["graph"]
    incoming = list(G.predecessors(node_id)) if node_id in G else []
    outgoing = list(G.successors(node_id)) if node_id in G else []

    return {
        **node,
        "grok_relevant": filter_result.pass_filter if filter_result else None,
        "grok_role": filter_result.potential_role if filter_result else None,
        "grok_reason": filter_result.reason if filter_result else None,
        "incoming_connections": len(incoming),
        "outgoing_connections": len(outgoing),
        "x_url": f"https://x.com/{node['handle']}",
    }


# --- Handle Submission Endpoints ---


def generate_submission_id() -> str:
    """Generate a unique submission ID."""
    return str(uuid.uuid4())[:8]


async def add_submission_to_graph(handle: str) -> Optional[dict]:
    """Add a submitted handle to the graph before evaluation."""
    handle = handle.strip().lstrip("@").lower()

    try:
        client = get_x_client()

        # Fetch user profile from X API
        user = client.get_user_by_username(handle)
        if not user:
            graph_state["last_error"] = f"User @{handle} not found"
            return None

        uid = user.get("id")
        if not uid:
            return None

        metrics = user.get("public_metrics", {})

        # Add to graph with "pending evaluation" metadata
        node_data = {
            "id": uid,
            "handle": user.get("username", handle),
            "name": user.get("name", ""),
            "bio": user.get("description", "")[:500],
            "followers_count": metrics.get("followers_count", 0),
            "following_count": metrics.get("following_count", 0),
            "tweet_count": metrics.get("tweet_count", 0),
            "is_seed": False,
            "is_candidate": True,
            "discovered_via": "user_submission",
            "grok_relevant": None,  # Pending evaluation
            "grok_role": None,
            "pagerank_score": 0.0,
            "underratedness_score": 0.0,
            "depth": 0,
            "submission_pending": True,
        }

        # Add to graph state
        graph_state["nodes"][uid] = node_data
        graph_state["graph"].add_node(uid, **node_data)

        # Recompute PageRank with new node
        compute_pagerank_scores()

        print(f"[submit] Added @{handle} to graph (ID: {uid})")
        return node_data

    except Exception as e:
        graph_state["last_error"] = f"Error adding @{handle}: {str(e)}"
        print(f"[submit] Error adding @{handle}: {e}")
        return None


async def process_submission(submission_id: str, handle: str):
    """Process submission through evaluation pipeline."""
    handle = handle.lower().lstrip("@")

    try:
        # Stage 1: Fast Screen
        db_update_submission_status(submission_id, "processing", "fast_screen")

        grok = get_grok_client()

        # Find node in graph by handle
        node = None
        node_id = None
        for nid, ndata in graph_state["nodes"].items():
            if ndata.get("handle", "").lower() == handle:
                node = ndata
                node_id = nid
                break

        if not node:
            db_update_submission_status(
                submission_id,
                "failed",
                "error",
                error_message=f"Node for @{handle} not found in graph",
            )
            return

        # Run fast screen
        fast_result = grok.fast_screen(
            handle=handle,
            bio=node.get("bio", ""),
            pinned_tweet=None,
            location=None,
        )

        # Update graph node with fast screen result
        node["grok_relevant"] = fast_result.pass_filter
        node["grok_role"] = fast_result.potential_role
        node["submission_pending"] = False
        graph_state["filter_results"][node_id] = fast_result

        # Save fast screen result to database
        fast_result_json = json.dumps(
            {
                "pass_filter": fast_result.pass_filter,
                "potential_role": fast_result.potential_role,
                "reason": fast_result.reason,
                "confidence": getattr(fast_result, "confidence", None),
            }
        )

        if not fast_result.pass_filter:
            db_update_submission_status(
                submission_id,
                "filtered_out",
                "done",
                fast_screen_result=fast_result_json,
            )
            print(f"[submit] @{handle} filtered out: {fast_result.reason}")
            return

        db_update_submission_status(
            submission_id,
            "processing",
            "fast_screen",
            fast_screen_result=fast_result_json,
        )

        # Stage 2: Deep Evaluation
        db_update_submission_status(submission_id, "processing", "deep_eval")

        # Get user's recent tweets for deep eval
        try:
            client = get_x_client()
            tweets = client.get_user_tweets(node_id, max_results=20)
        except Exception:
            tweets = []

        # Run deep evaluation
        deep_result = grok.deep_evaluate(
            handle=handle,
            bio=node.get("bio", ""),
            tweets=tweets,
            github_url=None,
        )

        # Update graph node with deep eval scores
        if deep_result:
            node["final_score"] = deep_result.get("final_score", 0)
            node["grok_evaluated"] = True

            # Save to enriched directory
            enriched_path = ENRICHED_DIR / f"deep_{handle}.json"
            enriched_path.parent.mkdir(parents=True, exist_ok=True)
            with open(enriched_path, "w") as f:
                json.dump(deep_result, f, indent=2)

            # Refresh evaluation cache
            global _evaluation_cache
            _evaluation_cache = None

        # Save deep eval result to database
        deep_eval_json = json.dumps(deep_result) if deep_result else None

        db_update_submission_status(
            submission_id, "completed", "done", deep_eval_result=deep_eval_json
        )

        # Recompute PageRank after evaluation
        compute_pagerank_scores()

        print(
            f"[submit] @{handle} evaluation complete: score={deep_result.get('final_score', 'N/A')}"
        )

    except Exception as e:
        print(f"[submit] Error processing @{handle}: {e}")
        db_update_submission_status(
            submission_id, "failed", "error", error_message=str(e)
        )


@app.post("/submit", response_model=SubmissionStatus)
async def submit_handle(
    request: HandleSubmitRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a handle for evaluation.
    The handle will be added to the graph immediately and processed in the background.
    """
    handle = request.handle.lower().lstrip("@").strip()

    if not handle:
        raise HTTPException(status_code=400, detail="Handle is required")

    # Check if already submitted
    existing = db_get_submission_by_handle(handle)
    if existing:
        # Return existing submission status
        position = get_submission_queue_position(existing["id"])
        return SubmissionStatus(
            submission_id=existing["id"],
            handle=existing["handle"],
            status=existing["status"],
            stage=existing["stage"],
            approval_status=existing["approval_status"],
            submitted_at=existing["submitted_at"],
            started_at=existing.get("started_at"),
            completed_at=existing.get("completed_at"),
            fast_screen_result=json.loads(existing["fast_screen_result"])
            if existing.get("fast_screen_result")
            else None,
            deep_eval_result=json.loads(existing["deep_eval_result"])
            if existing.get("deep_eval_result")
            else None,
            error=existing.get("error_message"),
            position_in_queue=position if existing["status"] == "pending" else None,
        )

    # Create submission record
    submission_id = generate_submission_id()
    submission = db_create_submission(submission_id, handle)

    # Immediately add to graph as pending node
    node = await add_submission_to_graph(handle)
    if not node:
        # Still create submission but note the error
        db_update_submission_status(
            submission_id,
            "failed",
            "fetching",
            error_message=graph_state.get("last_error", "Failed to fetch user profile"),
        )
        raise HTTPException(
            status_code=404, detail=f"Could not find user @{handle} on X"
        )

    # Auto-approve and start processing (no approval queue for now)
    db_update_submission_approval(submission_id, "approved")

    # Queue background evaluation via arq (Redis) or fallback to BackgroundTasks
    if arq_pool:
        try:
            await arq_pool.enqueue_job("process_submission", submission_id, handle)
            print(f"[submit] Queued job via arq: {submission_id}")
        except Exception as e:
            print(f"[submit] arq enqueue failed, falling back to BackgroundTasks: {e}")
            background_tasks.add_task(process_submission, submission_id, handle)
    else:
        background_tasks.add_task(process_submission, submission_id, handle)

    return SubmissionStatus(
        submission_id=submission_id,
        handle=handle,
        status="pending",
        stage="fetching",
        approval_status="approved",
        submitted_at=submission["submitted_at"],
    )


@app.get("/submit/{submission_id}", response_model=SubmissionStatus)
async def get_submission_status(submission_id: str):
    """Get status of a submission."""
    submission = db_get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    position = get_submission_queue_position(submission_id)

    return SubmissionStatus(
        submission_id=submission["id"],
        handle=submission["handle"],
        status=submission["status"],
        stage=submission["stage"],
        approval_status=submission["approval_status"],
        submitted_at=submission["submitted_at"],
        started_at=submission.get("started_at"),
        completed_at=submission.get("completed_at"),
        fast_screen_result=json.loads(submission["fast_screen_result"])
        if submission.get("fast_screen_result")
        else None,
        deep_eval_result=json.loads(submission["deep_eval_result"])
        if submission.get("deep_eval_result")
        else None,
        error=submission.get("error_message"),
        position_in_queue=position if submission["status"] == "pending" else None,
    )


@app.get("/admin/pending", response_model=List[PendingApproval])
async def get_pending_submissions(user: dict = Depends(require_admin)):
    """Get list of submissions pending approval (admin only)."""
    pending = db_get_pending_approvals()
    return [
        PendingApproval(
            submission_id=p["id"],
            handle=p["handle"],
            submitted_by=p.get("submitted_by"),
            submitted_at=p["submitted_at"],
        )
        for p in pending
    ]


@app.post("/admin/approve/{submission_id}")
async def approve_submission(
    submission_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    """Approve a pending submission and start processing."""
    submission = db_get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission["approval_status"] != "pending":
        raise HTTPException(status_code=400, detail="Submission already processed")

    # Update approval status
    db_update_submission_approval(submission_id, "approved")

    # Add to graph if not already there
    handle = submission["handle"]
    node_exists = any(
        n.get("handle", "").lower() == handle for n in graph_state["nodes"].values()
    )
    if not node_exists:
        await add_submission_to_graph(handle)

    # Queue background evaluation
    background_tasks.add_task(process_submission, submission_id, handle)

    return {"message": f"Submission {submission_id} approved", "handle": handle}


@app.post("/admin/reject/{submission_id}")
async def reject_submission(submission_id: str, user: dict = Depends(require_admin)):
    """Reject a pending submission."""
    submission = db_get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission["approval_status"] != "pending":
        raise HTTPException(status_code=400, detail="Submission already processed")

    db_update_submission_approval(submission_id, "rejected")

    return {
        "message": f"Submission {submission_id} rejected",
        "handle": submission["handle"],
    }


@app.get("/admin/submissions")
async def list_all_submissions(user: dict = Depends(require_admin)):
    """List all submissions."""
    submissions = db_get_all_submissions()
    return {"submissions": submissions, "count": len(submissions)}


@app.delete("/admin/submissions/failed")
async def clear_failed_submissions(user: dict = Depends(require_admin)):
    """Delete all failed submissions."""
    count = db_delete_failed_submissions()
    return {"message": f"Deleted {count} failed submissions", "deleted": count}


@app.delete("/admin/submissions/{submission_id}")
async def delete_submission(submission_id: str, user: dict = Depends(require_admin)):
    """Delete a specific submission."""
    deleted = db_delete_submission(submission_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {"message": f"Deleted submission {submission_id}"}


@app.post("/admin/submit-seed", response_model=SubmissionStatus)
async def submit_seed(
    request: SeedSubmitRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_admin),
):
    """
    Submit a seed account for full graph expansion pipeline (admin only).

    This triggers:
    1. Fetch X profile for the seed
    2. Build graph from seed's follows/retweets/replies
    3. Fast screen all discovered candidates
    4. Deep evaluation on top candidates
    5. Compute PageRank and underratedness scores
    6. Export results
    """
    handle = request.handle.lower().lstrip("@").strip()

    if not handle:
        raise HTTPException(status_code=400, detail="Handle is required")

    # Check if already submitted
    existing = db_get_submission_by_handle(handle)
    if existing and existing["status"] in ("pending", "processing"):
        raise HTTPException(
            status_code=409,
            detail=f"Seed @{handle} is already being processed (status: {existing['status']})"
        )

    # Create new submission with seed type
    submission_id = str(uuid.uuid4())[:8]
    db_create_submission(
        submission_id=submission_id,
        handle=handle,
        submitted_by=user.get("handle", "admin"),
    )

    print(f"[submit-seed] Created seed submission {submission_id} for @{handle}")

    # Queue the job via arq if Redis is available
    if arq_pool:
        try:
            await arq_pool.enqueue_job(
                "process_seed_submission",
                submission_id,
                handle,
                request.depth,
                request.max_following,
                request.max_candidates_to_eval,
            )
            print(f"[submit-seed] Queued job via arq: {submission_id}")
        except Exception as e:
            print(f"[submit-seed] arq enqueue failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue seed processing job: {e}"
            )
    else:
        raise HTTPException(
            status_code=503,
            detail="Redis worker not available. Please try again later."
        )

    return SubmissionStatus(
        submission_id=submission_id,
        handle=handle,
        status="pending",
        stage="queued",
        approval_status="approved",  # Admin submissions are auto-approved
        submitted_at=datetime.utcnow().isoformat(),
        queue_position=1,
    )


# --- Authentication Endpoints ---


def generate_pkce_pair():
    """Generate PKCE code verifier and challenge."""
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return code_verifier, code_challenge


@app.get("/auth/x/login")
async def x_login(request: Request):
    """Initiate X OAuth 2.0 login flow with PKCE."""
    if not X_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="X OAuth not configured. Set X_CLIENT_SECRET_ID environment variable.",
        )

    # Generate PKCE pair
    code_verifier, code_challenge = generate_pkce_pair()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)

    # Store state and verifier in Redis (shared across machines)
    store_oauth_state(
        state,
        {
            "code_verifier": code_verifier,
            "created_at": datetime.utcnow().isoformat(),
        },
    )

    # Build authorization URL
    auth_url = (
        "https://twitter.com/i/oauth2/authorize?"
        f"response_type=code"
        f"&client_id={X_CLIENT_ID}"
        f"&redirect_uri={quote(X_CALLBACK_URL)}"
        f"&scope=tweet.read%20users.read%20offline.access"
        f"&state={state}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=S256"
    )

    return RedirectResponse(url=auth_url)


@app.get("/auth/x/callback")
async def x_callback(code: str = None, state: str = None, error: str = None):
    """Handle X OAuth callback."""
    # Handle errors
    if error:
        return RedirectResponse(url=f"{FRONTEND_URL}?error={error}")

    if not code or not state:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=missing_params")

    # Verify state from Redis (shared across machines)
    state_data = get_oauth_state(state)
    if not state_data:
        return RedirectResponse(url=f"{FRONTEND_URL}?error=invalid_state")

    code_verifier = state_data["code_verifier"]

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient() as client:
            # Token request
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": X_CALLBACK_URL,
                "code_verifier": code_verifier,
            }

            # Use basic auth for confidential clients
            auth_header = base64.b64encode(
                f"{X_CLIENT_ID}:{X_CLIENT_SECRET}".encode()
            ).decode()

            response = await client.post(
                "https://api.twitter.com/2/oauth2/token",
                data=token_data,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            if response.status_code != 200:
                print(f"Token exchange failed: {response.text}")
                return RedirectResponse(
                    url=f"{FRONTEND_URL}?error=token_exchange_failed"
                )

            tokens = response.json()
            access_token = tokens["access_token"]

            # Get user info
            user_response = await client.get(
                "https://api.twitter.com/2/users/me",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"user.fields": "id,name,username,profile_image_url"},
            )

            if user_response.status_code != 200:
                print(f"User info failed: {user_response.text}")
                return RedirectResponse(url=f"{FRONTEND_URL}?error=user_info_failed")

            user_data = user_response.json()["data"]

            # Create JWT token
            jwt_payload = {
                "user_id": user_data["id"],
                "handle": user_data["username"],
                "name": user_data.get("name", user_data["username"]),
            }
            jwt_token = create_jwt_token(jwt_payload)

            # Redirect to frontend with token
            redirect_url = (
                f"{FRONTEND_URL}?"
                f"token={jwt_token}"
                f"&user_id={user_data['id']}"
                f"&handle={user_data['username']}"
                f"&name={quote(user_data.get('name', ''))}"
            )

            return RedirectResponse(url=redirect_url)

    except Exception as e:
        print(f"OAuth error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=oauth_error")


@app.get("/auth/me")
async def get_me(user: dict = Depends(require_auth)):
    """Get current authenticated user info."""
    return {
        "user_id": user.get("user_id"),
        "handle": user.get("handle"),
        "name": user.get("name"),
        "is_admin": user.get("handle", "").lower() in ADMIN_HANDLES,
    }


@app.post("/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    """Logout endpoint (client should clear local storage)."""
    return {"message": "Logged out successfully"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
