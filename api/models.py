"""
Pydantic models for the Grok Recruiter API
"""

from typing import List, Optional
from pydantic import BaseModel


class ScoreBreakdown(BaseModel):
    """Individual criterion score."""
    score: int  # 0-10
    evidence: str


class CandidateBase(BaseModel):
    """Base candidate info from graph."""
    handle: str
    name: str
    bio: str
    followers_count: int
    following_count: int
    tweet_count: int
    discovered_via: str
    pagerank_score: float
    underratedness_score: float


class CandidateDeepEval(BaseModel):
    """Candidate with deep evaluation scores."""
    handle: str
    name: str
    bio: str
    followers_count: int

    # Deep evaluation scores
    final_score: float  # 0-100
    technical_depth: ScoreBreakdown
    project_evidence: ScoreBreakdown
    mission_alignment: ScoreBreakdown
    exceptional_ability: ScoreBreakdown
    communication: ScoreBreakdown

    # Assessment
    summary: str
    strengths: List[str]
    concerns: List[str]
    recommended_role: str

    # External links
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    top_repos: Optional[List[str]] = None

    # Graph metrics
    pagerank_score: float = 0.0
    underratedness_score: float = 0.0
    discovered_via: str = ""


class CandidateListItem(BaseModel):
    """Candidate item for list view with score breakdown."""
    handle: str
    name: str
    bio: str
    followers_count: int
    final_score: float
    recommended_role: str
    summary: str
    strengths: List[str]
    pagerank_score: float
    underratedness_score: float
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None

    # Score breakdown with evidence (for detailed view)
    technical_depth: Optional[ScoreBreakdown] = None
    project_evidence: Optional[ScoreBreakdown] = None
    mission_alignment: Optional[ScoreBreakdown] = None
    exceptional_ability: Optional[ScoreBreakdown] = None
    communication: Optional[ScoreBreakdown] = None


class SearchRequest(BaseModel):
    """Natural language search request."""
    query: str
    limit: int = 30
    min_score: float = 0.0
    role_filter: Optional[str] = None  # research/engineering/infrastructure


class SearchResponse(BaseModel):
    """Search results response."""
    query: str
    total_results: int
    candidates: List[CandidateListItem]
    thinking: Optional[str] = None  # Grok's reasoning trace
    search_criteria: Optional[str] = None  # Extracted search criteria


class CandidateDetailResponse(BaseModel):
    """Full candidate detail response."""
    candidate: CandidateDeepEval
    match_explanation: Optional[str] = None


class StatsResponse(BaseModel):
    """Pipeline statistics."""
    total_nodes: int
    total_candidates: int
    fast_screened: int
    deep_evaluated: int
    seed_accounts: int


# --- Saved Candidates Models ---


class SaveCandidateRequest(BaseModel):
    """Request to save a candidate."""
    handle: str
    notes: Optional[str] = None


class SavedCandidateResponse(BaseModel):
    """Response for saved candidate."""
    handle: str
    saved_at: str
    notes: Optional[str] = None


class SavedCandidatesListResponse(BaseModel):
    """List of saved candidates with full details."""
    saved: List[CandidateListItem]
    total: int


# --- DM Generation Models ---


class DMGenerateRequest(BaseModel):
    """Request to generate personalized DM."""
    handle: str
    custom_context: Optional[str] = None  # Optional: Job description, team info
    tone: str = "professional"  # professional, casual, enthusiastic


class DMGenerateResponse(BaseModel):
    """Generated DM response."""
    handle: str
    message: str
    x_intent_url: str
    character_count: int


# --- Graph Visualization Models ---


class AddSourceRequest(BaseModel):
    """Request to add a source account to the graph."""
    handle: str
    depth: int = 1
    max_following: int = 100


class SeedSubmitRequest(BaseModel):
    """Request to submit a seed account for full graph expansion pipeline."""
    handle: str
    depth: int = 1
    max_following: int = 500
    max_candidates_to_eval: int = 50


class FilterRequest(BaseModel):
    """Request to filter nodes using Grok."""
    query: str  # Natural language query for Grok to filter by


class GraphNode(BaseModel):
    """Node in the social graph."""
    id: str
    handle: str
    name: str
    bio: str
    followers_count: int
    following_count: int
    is_seed: bool
    is_candidate: bool
    pagerank_score: float
    underratedness_score: float
    grok_relevant: Optional[bool] = None
    grok_role: Optional[str] = None
    depth: int = 0
    discovered_via: Optional[str] = None
    submission_pending: Optional[bool] = None


class GraphEdge(BaseModel):
    """Edge in the social graph."""
    source: str
    target: str
    weight: float
    interaction_type: str


class GraphResponse(BaseModel):
    """Graph data for visualization."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: dict


class GraphStatusResponse(BaseModel):
    """Graph loading status."""
    loading: bool
    node_count: int
    edge_count: int
    seed_count: int
    last_error: Optional[str]


# --- Handle Submission Models ---


class HandleSubmitRequest(BaseModel):
    """Request to submit a handle for evaluation."""
    handle: str


class SubmissionStatus(BaseModel):
    """Status of a handle submission."""
    submission_id: str
    handle: str
    status: str  # pending, processing, completed, filtered_out, failed
    stage: str  # pending, fetching, fast_screen, deep_eval, done
    approval_status: str  # pending, approved, rejected
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    submitted_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    fast_screen_result: Optional[dict] = None
    deep_eval_result: Optional[dict] = None
    error: Optional[str] = None
    position_in_queue: Optional[int] = None


class PendingApproval(BaseModel):
    """Pending submission awaiting approval."""
    submission_id: str
    handle: str
    submitted_by: Optional[str] = None
    submitted_at: str
