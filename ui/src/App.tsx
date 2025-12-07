import React, { useState, useEffect } from 'react';
import { SearchBar } from './components/SearchBar';
import { CandidateCard } from './components/CandidateCard';
import { fetchCandidates, searchCandidates, fetchStats } from './api/client';
import type { Candidate, Stats } from './api/client';

function App() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    loadCandidates();
    loadStats();
  }, []);

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
    try {
      const result = await searchCandidates(query, 30);
      setCandidates(result.candidates);
    } catch (e) {
      setError('Search failed. Please try again.');
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>🍀</span>
          <span style={styles.logoText}>Grok Recruiter</span>
        </div>
        <div style={styles.subtitle}>
          Taste-Graph Talent Discovery
        </div>
      </header>

      <main style={styles.main}>
        <SearchBar onSearch={handleSearch} isLoading={isLoading} />

        {searchQuery && (
          <div style={styles.searchInfo}>
            Showing results for: <strong>{searchQuery}</strong>
            <button
              onClick={() => {
                setSearchQuery('');
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

        {isLoading ? (
          <div style={styles.loading}>Loading candidates...</div>
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
              />
            ))}
          </div>
        ) : (
          <div style={styles.empty}>
            No candidates found. Try a different search query.
          </div>
        )}
      </main>

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
  },
  header: {
    padding: '24px 32px',
    borderBottom: '1px solid var(--border-color)',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  logoIcon: {
    fontSize: '24px',
  },
  logoText: {
    fontSize: '20px',
    fontWeight: 700,
  },
  subtitle: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    marginTop: '4px',
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
    backgroundColor: 'rgba(248, 81, 73, 0.1)',
    border: '1px solid var(--accent-red)',
    borderRadius: '8px',
    color: 'var(--accent-red)',
    marginBottom: '24px',
  },
  loading: {
    textAlign: 'center',
    padding: '48px',
    color: 'var(--text-secondary)',
  },
  results: {},
  resultsHeader: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    marginBottom: '16px',
  },
  empty: {
    textAlign: 'center',
    padding: '48px',
    color: 'var(--text-secondary)',
  },
  footer: {
    padding: '16px',
    textAlign: 'center',
    fontSize: '12px',
    color: 'var(--text-muted)',
    borderTop: '1px solid var(--border-color)',
  },
};

export default App;
