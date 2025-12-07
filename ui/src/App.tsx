import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation, useNavigate, Link } from 'react-router-dom';
import { SearchBar } from './components/SearchBar';
import { CandidateCard } from './components/CandidateCard';
import { ThinkingTrace } from './components/ThinkingTrace';
import { AboutPage } from './components/AboutPage';
import { SavedCandidatesView } from './components/SavedCandidatesView';
import { DMComposer } from './components/DMComposer';
import { GraphView } from './components/GraphView';
import { Login } from './components/Login';
import { AuthCallback } from './components/AuthCallback';
import { MySubmissionsView } from './components/MySubmissionsView';
import { AdminPanel } from './components/AdminPanel';
import { CandidateDetailView } from './components/CandidateDetailView';
import {
  fetchCandidates,
  searchCandidatesStream,
  fetchStats,
  fetchSavedHandles,
  saveCandidate,
  unsaveCandidate,
} from './api/client';
import type { Candidate, Stats } from './api/client';
import { getStoredAuth, isAdmin, type User } from './lib/auth';

function App() {
  const location = useLocation();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [thinking, setThinking] = useState<string | undefined>();
  const [searchCriteria, setSearchCriteria] = useState<string | undefined>();

  // Auth state
  const [user, setUser] = useState<User | null>(null);
  const [isAuthCallback, setIsAuthCallback] = useState(false);

  // State for saved candidates and DM
  const [savedHandles, setSavedHandles] = useState<Set<string>>(new Set());
  const [dmCandidate, setDmCandidate] = useState<Candidate | null>(null);
  const [selectedHandle, setSelectedHandle] = useState<string | null>(null);

  // Determine current view from route
  const currentPath = location.pathname;

  // Check for auth callback on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has('token') || params.has('error')) {
      setIsAuthCallback(true);
    } else {
      // Load stored auth
      const authState = getStoredAuth();
      if (authState.isAuthenticated && authState.user) {
        setUser(authState.user);
      }
    }
  }, []);

  // Load initial data
  useEffect(() => {
    loadCandidates();
    loadStats();
  }, []);

  // Load saved handles when user logs in
  useEffect(() => {
    if (user) {
      loadSavedHandles();
    } else {
      setSavedHandles(new Set());
    }
  }, [user]);

  const loadSavedHandles = async () => {
    try {
      const data = await fetchSavedHandles();
      setSavedHandles(new Set(data.handles.map(h => h.toLowerCase())));
    } catch (e) {
      console.error('Failed to load saved handles:', e);
    }
  };

  const handleToggleSave = async (handle: string) => {
    const lowerHandle = handle.toLowerCase();
    const wasSaved = savedHandles.has(lowerHandle);

    // Optimistic update
    setSavedHandles(prev => {
      const next = new Set(prev);
      if (wasSaved) {
        next.delete(lowerHandle);
      } else {
        next.add(lowerHandle);
      }
      return next;
    });

    try {
      if (wasSaved) {
        await unsaveCandidate(handle);
      } else {
        await saveCandidate(handle);
      }
    } catch (e) {
      // Rollback on error
      setSavedHandles(prev => {
        const next = new Set(prev);
        if (wasSaved) {
          next.add(lowerHandle);
        } else {
          next.delete(lowerHandle);
        }
        return next;
      });
      console.error('Failed to toggle save:', e);
    }
  };

  const loadCandidates = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchCandidates({ limit: 1000, sort_by: 'final_score' });
      setCandidates(data);
    } catch (e) {
      setError('Failed to load candidates. Make sure the API server is running.');
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const data = await fetchStats();
      setStats(data);
    } catch (e) {
      console.error('Failed to load stats:', e);
    }
  };

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setSearchQuery(query);
    setError(null);
    setThinking('');
    setSearchCriteria(undefined);
    setCandidates([]);

    try {
      await searchCandidatesStream(query, 30, {
        onThinking: (chunk) => {
          setThinking((prev) => (prev || '') + chunk);
        },
        onCandidates: (candidates, criteria, _total) => {
          setCandidates(candidates);
          setSearchCriteria(criteria);
        },
        onError: (message) => {
          setError(message);
        },
        onDone: () => {
          setIsLoading(false);
        },
      });
    } catch (e) {
      setError('Search failed. Please try again.');
      console.error(e);
      setIsLoading(false);
    }
  };

  const handleAuthSuccess = () => {
    const authState = getStoredAuth();
    if (authState.user) {
      setUser(authState.user);
    }
    setIsAuthCallback(false);
  };

  const handleAuthError = (error: string) => {
    console.error('Auth error:', error);
    setIsAuthCallback(false);
  };

  const handleLogout = () => {
    setUser(null);
    navigate('/');
  };

  // Show auth callback page when processing OAuth redirect
  if (isAuthCallback) {
    return <AuthCallback onAuthSuccess={handleAuthSuccess} onAuthError={handleAuthError} />;
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerTop}>
          <div style={styles.logoSection}>
            <div style={styles.logo}>
              <span style={styles.logoIcon}>✦</span>
              <span style={styles.logoText}>Grok Underrated Recruiter</span>
            </div>
            <div style={styles.subtitle}>
              Understand the universe of talent
            </div>
          </div>
          <div style={styles.headerButtons}>
            <Link
              to="/"
              style={{
                ...styles.tabButton,
                ...(currentPath === '/' ? styles.tabButtonActive : {}),
                textDecoration: 'none',
              }}
            >
              Search
            </Link>
            <Link
              to="/graph"
              style={{
                ...styles.tabButton,
                ...(currentPath === '/graph' ? styles.tabButtonActive : {}),
                textDecoration: 'none',
              }}
            >
              Graph
            </Link>
            {user && (
              <Link
                to="/saved"
                style={{
                  ...styles.tabButton,
                  ...(currentPath === '/saved' ? styles.tabButtonActive : {}),
                  textDecoration: 'none',
                }}
              >
                Saved {savedHandles.size > 0 && `(${savedHandles.size})`}
              </Link>
            )}
            {user && (
              <Link
                to="/submit"
                style={{
                  ...styles.tabButton,
                  ...(currentPath === '/submit' ? styles.tabButtonActive : {}),
                  textDecoration: 'none',
                }}
              >
                Submit
              </Link>
            )}
            {user && isAdmin(user) && (
              <Link
                to="/admin"
                style={{
                  ...styles.tabButton,
                  ...styles.adminTab,
                  ...(currentPath === '/admin' ? styles.tabButtonActive : {}),
                  textDecoration: 'none',
                }}
              >
                Admin
              </Link>
            )}
            <Link
              to="/about"
              style={{
                ...styles.aboutButton,
                textDecoration: 'none',
              }}
            >
              How it works
            </Link>
            <Login user={user} onLogout={handleLogout} />
          </div>
        </div>
      </header>

      <Routes>
        <Route path="/about" element={<AboutPage onClose={() => navigate('/')} />} />
        <Route
          path="/saved"
          element={
            user ? (
              <SavedCandidatesView
                onBack={() => navigate('/')}
                onComposeDM={(candidate) => setDmCandidate(candidate)}
                savedHandles={savedHandles}
                onToggleSave={handleToggleSave}
              />
            ) : (
              <div style={styles.main}>
                <div style={styles.empty}>Please log in to view saved candidates.</div>
              </div>
            )
          }
        />
        <Route path="/graph" element={<GraphView />} />
        <Route
          path="/submit"
          element={
            user ? (
              <MySubmissionsView onBack={() => navigate('/')} />
            ) : (
              <div style={styles.main}>
                <div style={styles.empty}>Please log in to submit handles.</div>
              </div>
            )
          }
        />
        <Route
          path="/admin"
          element={
            user && isAdmin(user) ? (
              <AdminPanel user={user} onBack={() => navigate('/')} />
            ) : (
              <div style={styles.main}>
                <div style={styles.empty}>Admin access required.</div>
              </div>
            )
          }
        />
        <Route
          path="/"
          element={
            <main style={styles.main}>
              <SearchBar onSearch={handleSearch} isLoading={isLoading} />

              {searchQuery && (
                <div style={styles.searchInfo}>
                  Showing results for: <strong>{searchQuery}</strong>
                  <button
                    onClick={() => {
                      setSearchQuery('');
                      setThinking(undefined);
                      setSearchCriteria(undefined);
                      loadCandidates();
                    }}
                    style={styles.clearButton}
                  >
                    Clear
                  </button>
                </div>
              )}

              {stats && (
                <div style={styles.stats}>
                  <span>{stats.total_nodes.toLocaleString()} accounts discovered</span>
                  <span>·</span>
                  <span>{stats.fast_screened.toLocaleString()} fast screened</span>
                  <span>·</span>
                  <span>{stats.deep_evaluated.toLocaleString()} deep evaluated</span>
                </div>
              )}

              {error && <div style={styles.error}>{error}</div>}

              {thinking && (
                <ThinkingTrace thinking={thinking} criteria={searchCriteria} isStreaming={isLoading} />
              )}

              {isLoading && !thinking ? (
                <div style={styles.loading}>
                  {searchQuery ? 'Starting search...' : 'Loading candidates...'}
                </div>
              ) : candidates.length > 0 ? (
                <div style={styles.results}>
                  <div style={styles.resultsHeader}>Found {candidates.length} candidates</div>
                  {candidates.map((candidate) => (
                    <CandidateCard
                      key={candidate.handle}
                      candidate={candidate}
                      onClick={() => setSelectedHandle(candidate.handle)}
                      isSaved={savedHandles.has(candidate.handle.toLowerCase())}
                      onToggleSave={user ? () => handleToggleSave(candidate.handle) : undefined}
                    />
                  ))}
                </div>
              ) : (
                <div style={styles.empty}>No candidates found. Try a different search query.</div>
              )}
            </main>
          }
        />
      </Routes>

      {dmCandidate && (
        <DMComposer
          candidate={dmCandidate}
          onClose={() => setDmCandidate(null)}
        />
      )}

      {selectedHandle && (
        <CandidateDetailView
          handle={selectedHandle}
          onClose={() => setSelectedHandle(null)}
          isSaved={savedHandles.has(selectedHandle.toLowerCase())}
          onToggleSave={user ? () => handleToggleSave(selectedHandle) : undefined}
        />
      )}

      <footer style={styles.footer}>
        Built for xAI Hackathon · Powered by Grok + X API
      </footer>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: 'var(--bg-primary)',
  },
  header: {
    padding: '24px 32px',
    borderBottom: '1px solid var(--border-color)',
    background: 'var(--bg-secondary)',
  },
  headerTop: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  logoSection: {},
  headerButtons: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
  },
  tabButton: {
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  tabButtonActive: {
    backgroundColor: 'var(--accent-primary)',
    color: '#000',
    borderColor: 'var(--accent-primary)',
  },
  adminTab: {
    color: 'var(--accent-orange)',
    borderColor: 'var(--accent-orange)',
  },
  savedButton: {
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  savedButtonActive: {
    backgroundColor: 'var(--accent-primary)',
    color: '#000',
    borderColor: 'var(--accent-primary)',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  aboutButton: {
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  logoIcon: {
    fontSize: '20px',
    color: 'var(--accent-primary)',
  },
  logoText: {
    fontSize: '20px',
    fontWeight: 600,
    letterSpacing: '-0.02em',
  },
  subtitle: {
    fontSize: '13px',
    color: 'var(--text-muted)',
    marginTop: '6px',
    letterSpacing: '0.02em',
  },
  main: {
    flex: 1,
    maxWidth: '800px',
    width: '100%',
    margin: '0 auto',
    padding: '24px',
  },
  searchInfo: {
    marginBottom: '16px',
    color: 'var(--text-secondary)',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  clearButton: {
    padding: '4px 8px',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  stats: {
    display: 'flex',
    gap: '12px',
    fontSize: '14px',
    color: 'var(--text-muted)',
    marginBottom: '24px',
  },
  error: {
    padding: '16px',
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    border: '1px solid var(--accent-red)',
    borderRadius: '12px',
    color: 'var(--accent-red)',
    marginBottom: '24px',
  },
  loading: {
    textAlign: 'center',
    padding: '64px',
    color: 'var(--text-muted)',
    fontSize: '15px',
  },
  results: {},
  resultsHeader: {
    fontSize: '13px',
    color: 'var(--text-muted)',
    marginBottom: '16px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  empty: {
    textAlign: 'center',
    padding: '64px',
    color: 'var(--text-muted)',
  },
  footer: {
    padding: '20px',
    textAlign: 'center',
    fontSize: '12px',
    color: 'var(--text-muted)',
    borderTop: '1px solid var(--border-color)',
    background: 'var(--bg-secondary)',
  },
};

export default App;
