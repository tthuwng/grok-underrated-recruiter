"""
FastAPI Backend for Grok Recruiter

Endpoints:
- GET /candidates - List candidates with optional NL search
- GET /candidates/{handle} - Get candidate detail
- GET /stats - Get pipeline statistics
- POST /search - Natural language search with re-ranking
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    CandidateDeepEval,
    CandidateDetailResponse,
    CandidateListItem,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

load_dotenv()

app = FastAPI(
    title="Grok Recruiter API",
    description="Talent discovery from X graph + Grok evaluation",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
PROCESSED_DIR = DATA_DIR / "processed"


def load_deep_evaluations() -> List[dict]:
    """Load all deep evaluation results from cache."""
    evaluations = []

    if not ENRICHED_DIR.exists():
        return evaluations

    for path in ENRICHED_DIR.glob("deep_*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                evaluations.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return evaluations


def load_graph_nodes() -> dict:
    """Load node data from processed CSV."""
    import csv

    nodes = {}
    nodes_path = PROCESSED_DIR / "nodes.csv"

    if not nodes_path.exists():
        return nodes

    with open(nodes_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = row.get("handle", "")
            if handle:
                nodes[handle.lower()] = row

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
    """Convert evaluation dict to CandidateListItem."""
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
    )


def evaluation_to_detail(eval_data: dict, node_data: dict = None) -> CandidateDeepEval:
    """Convert evaluation dict to CandidateDeepEval."""
    return CandidateDeepEval(
        handle=eval_data.get("handle", ""),
        name=node_data.get("name", "") if node_data else "",
        bio=eval_data.get("bio", ""),
        followers_count=eval_data.get("followers", 0),
        final_score=eval_data.get("final_score", 0.0),
        technical_depth=ScoreBreakdown(
            score=eval_data.get("technical_depth", {}).get("score", 0),
            evidence=eval_data.get("technical_depth", {}).get("evidence", ""),
        ),
        project_evidence=ScoreBreakdown(
            score=eval_data.get("project_evidence", {}).get("score", 0),
            evidence=eval_data.get("project_evidence", {}).get("evidence", ""),
        ),
        mission_alignment=ScoreBreakdown(
            score=eval_data.get("mission_alignment", {}).get("score", 0),
            evidence=eval_data.get("mission_alignment", {}).get("evidence", ""),
        ),
        exceptional_ability=ScoreBreakdown(
            score=eval_data.get("exceptional_ability", {}).get("score", 0),
            evidence=eval_data.get("exceptional_ability", {}).get("evidence", ""),
        ),
        communication=ScoreBreakdown(
            score=eval_data.get("communication", {}).get("score", 0),
            evidence=eval_data.get("communication", {}).get("evidence", ""),
        ),
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


@app.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest):
    """
    Natural language search for candidates.

    Uses Grok to understand the query and re-rank candidates by relevance.

    Example queries:
    - "Find post-training engineers with RLHF experience"
    - "ML researchers who have published papers"
    - "Systems engineers good at CUDA and distributed training"
    """
    evaluations = get_evaluations()
    nodes = get_nodes()

    query = request.query.lower()

    # Simple keyword matching for now
    # TODO: Use Grok to parse query and re-rank
    keywords = (
        query.replace("find", "")
        .replace("engineers", "")
        .replace("researchers", "")
        .split()
    )
    keywords = [k.strip() for k in keywords if len(k) > 2]

    # Score candidates by keyword match
    scored = []
    for e in evaluations:
        if request.min_score > 0 and e.get("final_score", 0) < request.min_score:
            continue
        if request.role_filter and e.get("recommended_role") != request.role_filter:
            continue

        # Match keywords against bio, skills, summary
        text = " ".join(
            [
                e.get("bio", ""),
                e.get("summary", ""),
                " ".join(e.get("strengths", [])),
                e.get("technical_depth", {}).get("evidence", ""),
                e.get("project_evidence", {}).get("evidence", ""),
            ]
        ).lower()

        match_score = sum(1 for kw in keywords if kw in text)
        if match_score > 0 or not keywords:
            scored.append((e, match_score + e.get("final_score", 0) / 100))

    # Sort by match score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Convert to response
    results = []
    for e, _ in scored[: request.limit]:
        handle = e.get("handle", "").lower()
        node = nodes.get(handle, {})
        results.append(evaluation_to_list_item(e, node))

    return SearchResponse(
        query=request.query,
        total_results=len(scored),
        candidates=results,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics."""
    evaluations = get_evaluations()
    nodes = get_nodes()

    # Count fast screens
    fast_screen_dir = DATA_DIR / "evaluations" / "fast_screen"
    fast_screened = (
        len(list(fast_screen_dir.glob("*.json"))) if fast_screen_dir.exists() else 0
    )

    # Count seeds
    seed_count = sum(1 for n in nodes.values() if n.get("is_root") == "True")

    return StatsResponse(
        total_nodes=len(nodes),
        total_candidates=sum(
            1 for n in nodes.values() if n.get("is_candidate") == "True"
        ),
        fast_screened=fast_screened,
        deep_evaluated=len(evaluations),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
