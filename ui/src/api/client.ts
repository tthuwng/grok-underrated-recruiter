// API client for Grok Recruiter

const API_BASE = '/api';

export interface ScoreBreakdown {
  score: number;
  evidence: string;
}

export interface Candidate {
  handle: string;
  name: string;
  bio: string;
  followers_count: number;
  final_score: number;
  recommended_role: string;
  summary: string;
  strengths: string[];
  pagerank_score: number;
  underratedness_score: number;
  github_url?: string;
  linkedin_url?: string;
}

export interface CandidateDetail extends Candidate {
  technical_depth: ScoreBreakdown;
  project_evidence: ScoreBreakdown;
  mission_alignment: ScoreBreakdown;
  exceptional_ability: ScoreBreakdown;
  communication: ScoreBreakdown;
  concerns: string[];
  top_repos?: string[];
  discovered_via: string;
}

export interface SearchResponse {
  query: string;
  total_results: number;
  candidates: Candidate[];
}

export interface Stats {
  total_nodes: number;
  total_candidates: number;
  fast_screened: number;
  deep_evaluated: number;
  seed_accounts: number;
}

export async function fetchCandidates(params?: {
  limit?: number;
  offset?: number;
  sort_by?: string;
  role?: string;
  min_score?: number;
}): Promise<Candidate[]> {
  const searchParams = new URLSearchParams();
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  if (params?.offset) searchParams.set('offset', params.offset.toString());
  if (params?.sort_by) searchParams.set('sort_by', params.sort_by);
  if (params?.role) searchParams.set('role', params.role);
  if (params?.min_score) searchParams.set('min_score', params.min_score.toString());

  const res = await fetch(`${API_BASE}/candidates?${searchParams}`);
  if (!res.ok) throw new Error('Failed to fetch candidates');
  return res.json();
}

export async function fetchCandidate(handle: string): Promise<{ candidate: CandidateDetail }> {
  const res = await fetch(`${API_BASE}/candidates/${handle}`);
  if (!res.ok) throw new Error('Candidate not found');
  return res.json();
}

export async function searchCandidates(query: string, limit = 30): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) throw new Error('Search failed');
  return res.json();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}
