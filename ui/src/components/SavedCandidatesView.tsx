import React, { useState, useEffect } from 'react';
import { CandidateCard } from './CandidateCard';
import type { Candidate } from '../api/client';
import { fetchSavedCandidates } from '../api/client';

interface SavedCandidatesViewProps {
  onBack: () => void;
  onComposeDM: (candidate: Candidate) => void;
  savedHandles: Set<string>;
  onToggleSave: (handle: string) => void;
}

export function SavedCandidatesView({
  onBack,
  onComposeDM,
  savedHandles,
  onToggleSave,
}: SavedCandidatesViewProps) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSaved();
  }, []);

  const loadSaved = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchSavedCandidates();
      setCandidates(data.saved);
    } catch (e) {
      setError('Failed to load saved candidates');
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemove = (handle: string) => {
    onToggleSave(handle);
    setCandidates(prev =>
      prev.filter(c => c.handle.toLowerCase() !== handle.toLowerCase())
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button onClick={onBack} style={styles.backButton}>
          ← Back to Search
        </button>
        <h2 style={styles.title}>Saved Candidates ({candidates.length})</h2>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {isLoading ? (
        <div style={styles.loading}>Loading saved candidates...</div>
      ) : candidates.length === 0 ? (
        <div style={styles.empty}>
          <div style={styles.emptyIcon}>♡</div>
          <p style={styles.emptyText}>No saved candidates yet.</p>
          <p style={styles.emptyHint}>
            Click the heart icon on any candidate card to save them here.
          </p>
        </div>
      ) : (
        <div style={styles.list}>
          {candidates.map(candidate => (
            <div key={candidate.handle} style={styles.cardWrapper}>
              <CandidateCard
                candidate={candidate}
                onClick={() => window.open(`https://x.com/${candidate.handle}`, '_blank')}
                isSaved={savedHandles.has(candidate.handle.toLowerCase())}
                onToggleSave={() => handleRemove(candidate.handle)}
              />
              <button
                style={styles.dmButton}
                onClick={(e) => {
                  e.stopPropagation();
                  onComposeDM(candidate);
                }}
              >
                Compose DM
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '24px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
    marginBottom: '24px',
  },
  backButton: {
    padding: '8px 16px',
    fontSize: '14px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
  },
  title: {
    fontSize: '20px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    margin: 0,
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
  empty: {
    textAlign: 'center',
    padding: '64px 24px',
  },
  emptyIcon: {
    fontSize: '64px',
    color: 'var(--text-muted)',
    marginBottom: '16px',
  },
  emptyText: {
    fontSize: '18px',
    color: 'var(--text-secondary)',
    margin: '0 0 8px 0',
  },
  emptyHint: {
    fontSize: '14px',
    color: 'var(--text-muted)',
    margin: 0,
  },
  list: {},
  cardWrapper: {
    position: 'relative',
    marginBottom: '16px',
  },
  dmButton: {
    position: 'absolute',
    top: '20px',
    right: '20px',
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: 500,
    color: '#000',
    backgroundColor: 'var(--accent-primary)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    zIndex: 10,
  },
};
