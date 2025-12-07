import React, { useState, useEffect } from 'react';
import { SearchBar } from './components/SearchBar';
import { CandidateCard } from './components/CandidateCard';
import { ThinkingTrace } from './components/ThinkingTrace';
import { AboutPage } from './components/AboutPage';
import { SavedCandidatesView } from './components/SavedCandidatesView';
import { DMComposer } from './components/DMComposer';
import { GraphView } from './components/GraphView';
import {
  fetchCandidates,
  searchCandidatesStream,
  fetchStats,
  fetchSavedHandles,
  saveCandidate,
  unsaveCandidate,
} from './api/client';
import type { Candidate, Stats } from './api/client';

function App() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [thinking, setThinking] = useState<string | undefined>();
  const [searchCriteria, setSearchCriteria] = useState<string | undefined>();
  const [showAbout, setShowAbout] = useState(false);

  // New state for saved candidates and DM
  const [currentView, setCurrentView] = useState<'search' | 'saved' | 'graph'>('search');
  const [savedHandles, setSavedHandles] = useState<Set<string>>(new Set());
  const [dmCandidate, setDmCandidate] = useState<Candidate | null>(null);

  // Load initial data
  useEffect(() => {
    loadCandidates();
    loadStats();
    loadSavedHandles();
  }, []);

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
      const data = await fetchCandidates({ limit: 30, sort_by: 'final_score' });
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
            <button
              style={{
                ...styles.tabButton,
                ...(currentView === 'search' ? styles.tabButtonActive : {}),
              }}
              onClick={() => setCurrentView('search')}
            >
              Search
            </button>
            <button
              style={{
                ...styles.tabButton,
                ...(currentView === 'graph' ? styles.tabButtonActive : {}),
              }}
              onClick={() => setCurrentView('graph')}
            >
              Graph
            </button>
            <button
              style={{
                ...styles.tabButton,
                ...(currentView === 'saved' ? styles.tabButtonActive : {}),
              }}
              onClick={() => setCurrentView('saved')}
            >
              Saved {savedHandles.size > 0 && `(${savedHandles.size})`}
            </button>
            <button
              style={styles.aboutButton}
              onClick={() => setShowAbout(true)}
            >
              How it works
            </button>
          </div>
        </div>
      </header>

      {showAbout && <AboutPage onClose={() => setShowAbout(false)} />}

      {currentView === 'saved' ? (
        <SavedCandidatesView
          onBack={() => setCurrentView('search')}
          onComposeDM={(candidate) => setDmCandidate(candidate)}
          savedHandles={savedHandles}
          onToggleSave={handleToggleSave}
        />
      ) : currentView === 'graph' ? (
        <GraphView />
      ) : (
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

          {error && (
            <div style={styles.error}>{error}</div>
          )}

          {thinking && (
            <ThinkingTrace thinking={thinking} criteria={searchCriteria} isStreaming={isLoading} />
          )}

          {isLoading && !thinking ? (
            <div style={styles.loading}>
              {searchQuery ? 'Starting search...' : 'Loading candidates...'}
            </div>
          ) : candidates.length > 0 ? (
            <div style={styles.results}>
              <div style={styles.resultsHeader}>
                Found {candidates.length} candidates
              </div>
              {candidates.map((candidate) => (
                <CandidateCard
                  key={candidate.handle}
                  candidate={candidate}
                  onClick={() => {
                    window.open(`https://x.com/${candidate.handle}`, '_blank');
                  }}
                  isSaved={savedHandles.has(candidate.handle.toLowerCase())}
                  onToggleSave={() => handleToggleSave(candidate.handle)}
                />
              ))}
            </div>
          ) : (
            <div style={styles.empty}>
              No candidates found. Try a different search query.
            </div>
          )}
        </main>
      )}

      {dmCandidate && (
        <DMComposer
          candidate={dmCandidate}
          onClose={() => setDmCandidate(null)}
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
