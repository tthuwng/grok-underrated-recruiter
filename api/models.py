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
    """Candidate item for list view (minimal)."""
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
