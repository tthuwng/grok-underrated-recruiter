/**
 * Authentication utilities for Grok Underrated Recruiter
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';
const TOKEN_KEY = 'grok_recruiter_token';
const USER_KEY = 'grok_recruiter_user';

export interface User {
  user_id: string;
  x_handle: string;
  x_name?: string;
  x_profile_image_url?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
}

/**
 * Get stored auth state from localStorage
 */
export function getStoredAuth(): AuthState {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const userStr = localStorage.getItem(USER_KEY);
    const user = userStr ? JSON.parse(userStr) : null;

    return {
      isAuthenticated: !!token && !!user,
      user,
      token,
    };
  } catch {
    return { isAuthenticated: false, user: null, token: null };
  }
}

/**
 * Store auth state in localStorage
 */
export function setStoredAuth(token: string, user: User): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

/**
 * Clear stored auth state
 */
export function clearStoredAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

/**
 * Get the current auth token
 */
export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Initiate X OAuth login
 */
export function loginWithX(): void {
  // Redirect to backend OAuth endpoint
  window.location.href = `${API_BASE}/auth/x/login`;
}

/**
 * Handle OAuth callback - parse URL params and store auth
 */
export function handleAuthCallback(): AuthState | null {
  const params = new URLSearchParams(window.location.search);

  const token = params.get('token');
  const userId = params.get('user_id');
  const handle = params.get('handle');
  const name = params.get('name');
  const error = params.get('error');

  if (error) {
    console.error('Auth error:', error);
    return null;
  }

  if (token && userId && handle) {
    const user: User = {
      user_id: userId,
      x_handle: handle,
      x_name: name || handle,
    };

    setStoredAuth(token, user);

    // Clean up URL
    window.history.replaceState({}, '', window.location.pathname);

    return {
      isAuthenticated: true,
      user,
      token,
    };
  }

  return null;
}

/**
 * Logout - clear stored auth and notify backend
 */
export async function logout(): Promise<void> {
  const token = getAuthToken();

  if (token) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
    } catch {
      // Ignore errors, just clear local state
    }
  }

  clearStoredAuth();
}

/**
 * Verify current token is still valid
 */
export async function verifyAuth(): Promise<boolean> {
  const token = getAuthToken();

  if (!token) {
    return false;
  }

  try {
    const response = await fetch(`${API_BASE}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (response.ok) {
      return true;
    } else {
      // Token is invalid, clear it
      clearStoredAuth();
      return false;
    }
  } catch {
    return false;
  }
}

/**
 * Get Authorization header for API requests
 */
export function getAuthHeader(): Record<string, string> {
  const token = getAuthToken();
  if (token) {
    return { 'Authorization': `Bearer ${token}` };
  }
  return {};
}

/**
 * Admin handles - users who can approve/reject submissions
 */
const ADMIN_HANDLES = ['huwng_tran', 'ellenjxu_'];

/**
 * Check if a user is an admin
 */
export function isAdmin(user: User | null): boolean {
  if (!user) return false;
  return ADMIN_HANDLES.includes(user.x_handle.toLowerCase());
}

/**
 * Local storage key for user's submission IDs
 */
const SUBMISSIONS_KEY = 'grok_recruiter_submissions';

/**
 * Get stored submission IDs
 */
export function getStoredSubmissions(): string[] {
  try {
    const stored = localStorage.getItem(SUBMISSIONS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/**
 * Add a submission ID to local storage
 */
export function addStoredSubmission(submissionId: string): void {
  const submissions = getStoredSubmissions();
  if (!submissions.includes(submissionId)) {
    submissions.unshift(submissionId); // Add to front
    localStorage.setItem(SUBMISSIONS_KEY, JSON.stringify(submissions.slice(0, 50))); // Keep max 50
  }
}

/**
 * Remove a submission ID from local storage
 */
export function removeStoredSubmission(submissionId: string): void {
  const submissions = getStoredSubmissions();
  const filtered = submissions.filter(id => id !== submissionId);
  localStorage.setItem(SUBMISSIONS_KEY, JSON.stringify(filtered));
}
