import React, { useState, useEffect } from 'react';
import { fetchCandidate, type CandidateDetail, type ScoreBreakdown } from '../api/client';

interface CandidateDetailViewProps {
  handle: string;
  onClose: () => void;
  isSaved?: boolean;
  onToggleSave?: () => void;
}

export function CandidateDetailView({ handle, onClose, isSaved, onToggleSave }: CandidateDetailViewProps) {
  const [candidate, setCandidate] = useState<CandidateDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCandidate();
  }, [handle]);

  const loadCandidate = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { candidate } = await fetchCandidate(handle);
      setCandidate(candidate);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load candidate');
    } finally {
      setIsLoading(false);
    }
  };

  const renderScoreCard = (title: string, breakdown: ScoreBreakdown | undefined) => {
    // Skip if no breakdown or if it has no meaningful data (score 0 and no evidence)
    if (!breakdown || (breakdown.score === 0 && !breakdown.evidence)) return null;

    const scoreColor = breakdown.score >= 8 ? 'var(--accent-green)'
      : breakdown.score >= 6 ? 'var(--accent-primary)'
      : breakdown.score >= 4 ? 'var(--accent-orange)'
      : 'var(--text-muted)';

    return (
      <div style={styles.scoreCard}>
        <div style={styles.scoreHeader}>
          <span style={styles.scoreTitle}>{title}</span>
          <span style={{ ...styles.scoreValue, color: scoreColor }}>{breakdown.score}/10</span>
        </div>
        {breakdown.evidence && <p style={styles.scoreEvidence}>{breakdown.evidence}</p>}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div style={styles.overlay} onClick={onClose}>
        <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
          <div style={styles.loading}>Loading candidate details...</div>
        </div>
      </div>
    );
  }

  if (error || !candidate) {
    return (
      <div style={styles.overlay} onClick={onClose}>
        <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
          <div style={styles.error}>{error || 'Candidate not found'}</div>
          <button style={styles.closeButton} onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  const finalScoreColor = candidate.final_score >= 80 ? 'var(--accent-green)'
    : candidate.final_score >= 60 ? 'var(--accent-primary)'
    : candidate.final_score >= 40 ? 'var(--accent-orange)'
    : 'var(--text-muted)';

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <button style={styles.closeX} onClick={onClose}>×</button>

        <div style={styles.header}>
          <div style={styles.profileSection}>
            <div style={styles.nameRow}>
              <h1 style={styles.name}>{candidate.name}</h1>
              <a
                href={`https://x.com/${candidate.handle}`}
                target="_blank"
                rel="noopener noreferrer"
                style={styles.handle}
              >
                @{candidate.handle}
              </a>
            </div>
            <p style={styles.bio}>{candidate.bio}</p>
            <div style={styles.metaRow}>
              <span style={styles.metaItem}>{candidate.followers_count.toLocaleString()} followers</span>
              {candidate.discovered_via && (
                <span style={styles.metaItem}>Discovered via: {candidate.discovered_via}</span>
              )}
            </div>
          </div>

          <div style={styles.scoreSection}>
            <div style={styles.mainScore}>
              <span style={{ ...styles.mainScoreValue, color: finalScoreColor }}>
                {candidate.final_score.toFixed(0)}
              </span>
              <span style={styles.mainScoreLabel}>Final Score</span>
            </div>
          </div>
        </div>

        <div style={styles.roleBadge}>
          {candidate.recommended_role}
        </div>

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Summary</h2>
          <p style={styles.summary}>{candidate.summary}</p>
        </div>

        {candidate.strengths && candidate.strengths.length > 0 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Strengths</h2>
            <ul style={styles.list}>
              {candidate.strengths.map((strength, i) => (
                <li key={i} style={styles.listItem}>{strength}</li>
              ))}
            </ul>
          </div>
        )}

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Score Breakdown</h2>
          <div style={styles.scoresGrid}>
            {renderScoreCard('Technical Depth', candidate.technical_depth)}
            {renderScoreCard('Project Evidence', candidate.project_evidence)}
            {renderScoreCard('Mission Alignment', candidate.mission_alignment)}
            {renderScoreCard('Exceptional Ability', candidate.exceptional_ability)}
            {renderScoreCard('Communication', candidate.communication)}
          </div>
          {/* Show fallback if no score cards rendered */}
          {!candidate.technical_depth?.evidence &&
           !candidate.project_evidence?.evidence &&
           !candidate.mission_alignment?.evidence &&
           !candidate.exceptional_ability?.evidence &&
           !candidate.communication?.evidence && (
            <p style={styles.noScoresText}>
              Detailed score breakdown not available for this candidate.
            </p>
          )}
        </div>

        {candidate.concerns && candidate.concerns.length > 0 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Potential Concerns</h2>
            <ul style={styles.list}>
              {candidate.concerns.map((concern, i) => (
                <li key={i} style={styles.concernItem}>{concern}</li>
              ))}
            </ul>
          </div>
        )}

        {candidate.top_repos && candidate.top_repos.length > 0 && (
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Notable Repositories</h2>
            <div style={styles.repoList}>
              {candidate.top_repos.map((repo, i) => (
                <a
                  key={i}
                  href={repo.startsWith('http') ? repo : `https://github.com/${repo}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={styles.repoLink}
                >
                  {repo}
                </a>
              ))}
            </div>
          </div>
        )}

        <div style={styles.graphScores}>
          <div style={styles.graphScoreItem}>
            <span style={styles.graphScoreLabel}>PageRank</span>
            <span style={styles.graphScoreValue}>{candidate.pagerank_score.toFixed(4)}</span>
          </div>
          <div style={styles.graphScoreItem}>
            <span style={styles.graphScoreLabel}>Underratedness</span>
            <span style={styles.graphScoreValue}>{candidate.underratedness_score.toFixed(4)}</span>
          </div>
        </div>

        <div style={styles.actions}>
          <a
            href={`https://x.com/${candidate.handle}`}
            target="_blank"
            rel="noopener noreferrer"
            style={styles.viewProfileButton}
          >
            View on X
          </a>
          {candidate.github_url && (
            <a
              href={candidate.github_url}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.secondaryButton}
            >
              GitHub
            </a>
          )}
          {candidate.linkedin_url && (
            <a
              href={candidate.linkedin_url}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.secondaryButton}
            >
              LinkedIn
            </a>
          )}
          {onToggleSave && (
            <button
              onClick={onToggleSave}
              style={isSaved ? styles.unsaveButton : styles.saveButton}
            >
              {isSaved ? 'Unsave' : 'Save'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: '20px',
    overflowY: 'auto',
  },
  modal: {
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '16px',
    border: '1px solid var(--border-color)',
    maxWidth: '700px',
    width: '100%',
    maxHeight: '90vh',
    overflowY: 'auto',
    padding: '32px',
    position: 'relative',
  },
  closeX: {
    position: 'absolute',
    top: '16px',
    right: '16px',
    width: '32px',
    height: '32px',
    border: 'none',
    background: 'transparent',
    color: 'var(--text-muted)',
    fontSize: '24px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loading: {
    padding: '48px',
    textAlign: 'center',
    color: 'var(--text-muted)',
  },
  error: {
    padding: '48px',
    textAlign: 'center',
    color: 'var(--accent-red)',
  },
  closeButton: {
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--text-primary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
    marginTop: '16px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '24px',
    marginBottom: '20px',
  },
  profileSection: {
    flex: 1,
  },
  nameRow: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '12px',
    marginBottom: '8px',
    flexWrap: 'wrap',
  },
  name: {
    fontSize: '24px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    margin: 0,
  },
  handle: {
    fontSize: '16px',
    color: 'var(--accent-primary)',
    textDecoration: 'none',
  },
  bio: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
    margin: '0 0 12px 0',
  },
  metaRow: {
    display: 'flex',
    gap: '16px',
    flexWrap: 'wrap',
  },
  metaItem: {
    fontSize: '13px',
    color: 'var(--text-muted)',
  },
  scoreSection: {},
  mainScore: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '16px 24px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
  },
  mainScoreValue: {
    fontSize: '36px',
    fontWeight: 700,
  },
  mainScoreLabel: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  roleBadge: {
    display: 'inline-block',
    padding: '6px 14px',
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--accent-primary)',
    backgroundColor: 'rgba(96, 165, 250, 0.15)',
    borderRadius: '6px',
    marginBottom: '24px',
  },
  section: {
    marginBottom: '24px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: 600,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '12px',
  },
  summary: {
    fontSize: '15px',
    color: 'var(--text-primary)',
    lineHeight: 1.6,
    margin: 0,
  },
  list: {
    margin: 0,
    paddingLeft: '20px',
  },
  listItem: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    marginBottom: '6px',
    lineHeight: 1.5,
  },
  concernItem: {
    fontSize: '14px',
    color: 'var(--accent-orange)',
    marginBottom: '6px',
    lineHeight: 1.5,
  },
  scoresGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '12px',
  },
  scoreCard: {
    padding: '14px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '10px',
    border: '1px solid var(--border-color)',
  },
  scoreHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  scoreTitle: {
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-primary)',
  },
  scoreValue: {
    fontSize: '15px',
    fontWeight: 600,
  },
  scoreEvidence: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.5,
    margin: 0,
  },
  noScoresText: {
    fontSize: '14px',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
    marginTop: '8px',
  },
  repoList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
  },
  repoLink: {
    padding: '6px 12px',
    fontSize: '13px',
    color: 'var(--text-secondary)',
    backgroundColor: 'var(--bg-tertiary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    textDecoration: 'none',
  },
  graphScores: {
    display: 'flex',
    gap: '24px',
    padding: '16px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '10px',
    marginBottom: '24px',
  },
  graphScoreItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  graphScoreLabel: {
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  graphScoreValue: {
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    fontFamily: 'monospace',
  },
  actions: {
    display: 'flex',
    gap: '12px',
    flexWrap: 'wrap',
  },
  viewProfileButton: {
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: 500,
    color: '#000',
    backgroundColor: 'var(--accent-primary)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    textDecoration: 'none',
  },
  secondaryButton: {
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    cursor: 'pointer',
    textDecoration: 'none',
  },
  saveButton: {
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--accent-green)',
    backgroundColor: 'transparent',
    border: '1px solid var(--accent-green)',
    borderRadius: '8px',
    cursor: 'pointer',
  },
  unsaveButton: {
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--accent-red)',
    backgroundColor: 'transparent',
    border: '1px solid var(--accent-red)',
    borderRadius: '8px',
    cursor: 'pointer',
  },
};
