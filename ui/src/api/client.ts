// API client for Grok Underrated Recruiter

const API_BASE = import.meta.env.VITE_API_URL || '/api';

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
  // Score breakdown with evidence
  technical_depth?: ScoreBreakdown;
  project_evidence?: ScoreBreakdown;
  mission_alignment?: ScoreBreakdown;
  exceptional_ability?: ScoreBreakdown;
  communication?: ScoreBreakdown;
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
  thinking?: string;
  search_criteria?: string;
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

export async function searchCandidates(query: string, limit = 100): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) throw new Error('Search failed');
  return res.json();
}

export interface StreamCallbacks {
  onThinking: (chunk: string) => void;
  onCandidates: (candidates: Candidate[], criteria: string, total: number) => void;
  onError: (error: string) => void;
  onDone: () => void;
}

export async function searchCandidatesStream(
  query: string,
  limit: number,
  callbacks: StreamCallbacks
): Promise<void> {
  const res = await fetch(`${API_BASE}/search/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });

  if (!res.ok) {
    callbacks.onError('Search failed');
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);

      if (data === '[DONE]') {
        callbacks.onDone();
        return;
      }

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'thinking') {
          callbacks.onThinking(parsed.content);
        } else if (parsed.type === 'candidates') {
          callbacks.onCandidates(parsed.candidates, parsed.criteria, parsed.total);
        } else if (parsed.type === 'error') {
          callbacks.onError(parsed.message);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }
  }

  callbacks.onDone();
}

export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

// --- Saved Candidates API ---

export interface SavedCandidatesResponse {
  saved: Candidate[];
  total: number;
}

// Helper to get auth headers
function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('grok_recruiter_token');
  if (token) {
    return { 'Authorization': `Bearer ${token}` };
  }
  return {};
}

export async function saveCandidate(handle: string): Promise<void> {
  const res = await fetch(`${API_BASE}/saved`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ handle }),
  });
  if (!res.ok) {
    if (res.status === 401) throw new Error('Please log in to save candidates');
    if (res.status === 409) throw new Error('Already saved');
    throw new Error('Failed to save candidate');
  }
}

export async function unsaveCandidate(handle: string): Promise<void> {
  const res = await fetch(`${API_BASE}/saved/${handle}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    if (res.status === 401) throw new Error('Please log in to unsave candidates');
    throw new Error('Failed to unsave candidate');
  }
}

export async function fetchSavedCandidates(): Promise<SavedCandidatesResponse> {
  const res = await fetch(`${API_BASE}/saved`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch saved candidates');
  return res.json();
}

export async function fetchSavedHandles(): Promise<{ handles: string[] }> {
  const res = await fetch(`${API_BASE}/saved/handles`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch saved handles');
  return res.json();
}

export async function fetchSavedCount(): Promise<{ count: number }> {
  const res = await fetch(`${API_BASE}/saved/count`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch saved count');
  return res.json();
}

// --- DM Generation API ---

export interface DMStreamCallbacks {
  onContent: (content: string) => void;
  onDone: (message: string, xIntentUrl: string, charCount: number) => void;
  onError: (error: string) => void;
}

export async function generateDMStream(
  handle: string,
  customContext: string,
  tone: string,
  callbacks: DMStreamCallbacks
): Promise<void> {
  const res = await fetch(`${API_BASE}/dm/generate/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ handle, custom_context: customContext, tone }),
  });

  if (!res.ok) {
    callbacks.onError('DM generation failed');
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);

      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'content') {
          callbacks.onContent(parsed.content);
        } else if (parsed.type === 'done') {
          callbacks.onDone(parsed.message, parsed.x_intent_url, parsed.character_count);
        } else if (parsed.type === 'error') {
          callbacks.onError(parsed.message);
        }
      } catch {
        // Skip invalid JSON
      }
    }
  }
}

// --- Handle Submission API ---

export interface SubmissionStatus {
  submission_id: string;
  handle: string;
  status: string;
  stage: string;
  approval_status: string;
  submitted_at: string;
  started_at?: string;
  completed_at?: string;
  submitted_by?: string;
  approved_by?: string;
  approved_at?: string;
  fast_screen_result?: object;
  deep_eval_result?: object;
  error?: string;
  position_in_queue?: number;
}

export interface PendingApproval {
  submission_id: string;
  handle: string;
  submitted_by?: string;
  submitted_at: string;
}

export async function submitHandle(handle: string, token?: string): Promise<SubmissionStatus> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/submit`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ handle }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Submission failed' }));
    throw new Error(error.detail || 'Submission failed');
  }
  return res.json();
}

export async function getSubmissionStatus(submissionId: string): Promise<SubmissionStatus> {
  const res = await fetch(`${API_BASE}/submit/${submissionId}`);
  if (!res.ok) {
    throw new Error('Failed to get submission status');
  }
  return res.json();
}

export async function getPendingApprovals(token: string): Promise<PendingApproval[]> {
  const res = await fetch(`${API_BASE}/admin/pending`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 403) throw new Error('Not authorized');
    throw new Error('Failed to get pending approvals');
  }
  return res.json();
}

export async function approveSubmission(submissionId: string, token: string): Promise<void> {
  const res = await fetch(`${API_BASE}/admin/approve/${submissionId}`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 403) throw new Error('Not authorized');
    throw new Error('Failed to approve submission');
  }
}

export async function rejectSubmission(submissionId: string, token: string): Promise<void> {
  const res = await fetch(`${API_BASE}/admin/reject/${submissionId}`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 403) throw new Error('Not authorized');
    throw new Error('Failed to reject submission');
  }
}
