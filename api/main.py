"""
FastAPI Backend for Grok Recruiter

Endpoints:
- GET /candidates - List candidates with optional NL search
- GET /candidates/{handle} - Get candidate detail
- GET /stats - Get pipeline statistics
- POST /search - Natural language search with re-ranking
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    CandidateDeepEval,
    CandidateDetailResponse,
    CandidateListItem,
    DMGenerateRequest,
    DMGenerateResponse,
    SaveCandidateRequest,
    SavedCandidateResponse,
    SavedCandidatesListResponse,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from api.database import (
    init_db,
    save_candidate as db_save_candidate,
    unsave_candidate as db_unsave_candidate,
    get_saved_handles as db_get_saved_handles,
    get_saved_candidates as db_get_saved_candidates,
    is_candidate_saved,
    get_saved_count,
    save_dm_history,
)

load_dotenv()

app = FastAPI(
    title="Grok Underrated Recruiter API",
    description="Talent discovery from X graph + Grok evaluation",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


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
                    {"role": "system", "content": "You are an expert technical recruiter."},
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
        f"@{e.get('handle', '')}: {e.get('summary', '')[:150]}"
        for e in filtered[:50]
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
        text = " ".join([
            e.get("bio", ""),
            e.get("summary", ""),
            " ".join(e.get("strengths", [])),
            e.get("technical_depth", {}).get("evidence", ""),
        ]).lower()
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


# --- Saved Candidates Endpoints ---


@app.post("/saved", response_model=SavedCandidateResponse)
async def save_candidate(request: SaveCandidateRequest):
    """Save a candidate to the database."""
    handle = request.handle.lower().lstrip("@")

    # Verify candidate exists
    evaluations = get_evaluations()
    exists = any(e.get("handle", "").lower() == handle for e in evaluations)
    if not exists:
        raise HTTPException(status_code=404, detail=f"Candidate @{handle} not found")

    try:
        result = db_save_candidate(handle, request.notes)
        return SavedCandidateResponse(**result)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Candidate already saved")


@app.delete("/saved/{handle}")
async def unsave_candidate(handle: str):
    """Remove a candidate from saved list."""
    handle = handle.lower().lstrip("@")

    if not db_unsave_candidate(handle):
        raise HTTPException(status_code=404, detail="Candidate not in saved list")

    return {"status": "ok", "handle": handle}


@app.get("/saved", response_model=SavedCandidatesListResponse)
async def list_saved_candidates():
    """Get all saved candidates with their full evaluation data."""
    evaluations = get_evaluations()
    nodes = get_nodes()

    saved_handles = set(db_get_saved_handles())

    # Get full candidate data for saved handles
    results = []
    for e in evaluations:
        handle = e.get("handle", "").lower()
        if handle in saved_handles:
            node = nodes.get(handle, {})
            results.append(evaluation_to_list_item(e, node))

    return SavedCandidatesListResponse(saved=results, total=len(results))


@app.get("/saved/handles")
async def get_saved_handles():
    """Get list of all saved candidate handles."""
    handles = db_get_saved_handles()
    return {"handles": handles}


@app.get("/saved/count")
async def saved_count():
    """Get count of saved candidates."""
    count = get_saved_count()
    return {"count": count}


@app.get("/saved/check/{handle}")
async def check_if_saved(handle: str):
    """Check if a specific candidate is saved."""
    handle = handle.lower().lstrip("@")
    is_saved = is_candidate_saved(handle)
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
        "enthusiastic": "Use an enthusiastic, energetic tone. Show genuine excitement about their work."
    }

    # Build score details if available
    score_details = []
    for criterion in ['technical_depth', 'project_evidence', 'mission_alignment', 'exceptional_ability', 'communication']:
        if criterion in eval_data and eval_data[criterion]:
            data = eval_data[criterion]
            if isinstance(data, dict):
                score_details.append(f"  - {criterion.replace('_', ' ').title()}: {data.get('score', 'N/A')}/10 - {data.get('evidence', '')}")

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
- Handle: @{eval_data.get('handle', '')}
- Name: {eval_data.get('name', '')}
- Bio: {eval_data.get('bio', '')}
- Followers: {eval_data.get('followers_count', 0):,}
- Recommended Role: {eval_data.get('recommended_role', '')}
- GitHub: {eval_data.get('github_url', 'Not available')}
- LinkedIn: {eval_data.get('linkedin_url', 'Not available')}

EVALUATION SUMMARY:
{eval_data.get('summary', '')}

KEY STRENGTHS:
{chr(10).join('- ' + s for s in eval_data.get('strengths', [])[:4])}

DETAILED SCORES:
{score_section}
{context_section}
TONE: {tone_instructions.get(tone, tone_instructions['professional'])}

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
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        full_message += content
                        yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                except json.JSONDecodeError:
                    continue

            # Save to history
            save_dm_history(handle, request.custom_context, full_message)

            # Send final result with X intent URL
            x_dm_url = f"https://twitter.com/messages/compose?text={quote(full_message)}"
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
