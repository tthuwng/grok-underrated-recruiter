import React from 'react';
import type { Candidate } from '../api/client';

interface CandidateCardProps {
  candidate: Candidate;
  onClick?: () => void;
}

function ScoreDot({ score, maxScore = 10 }: { score: number; maxScore?: number }) {
  const percentage = (score / maxScore) * 100;
  let color = 'var(--accent-red)';
  if (percentage >= 70) color = 'var(--accent-green)';
  else if (percentage >= 50) color = 'var(--accent-orange)';

  return (
    <span
      style={{
        display: 'inline-block',
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        backgroundColor: color,
        marginRight: '4px',
      }}
    />
  );
}

export function CandidateCard({ candidate, onClick }: CandidateCardProps) {
  const roleColors: Record<string, string> = {
    research: 'var(--accent-purple)',
    engineering: 'var(--accent-blue)',
    infrastructure: 'var(--accent-orange)',
    none: 'var(--text-muted)',
  };

  return (
    <div style={styles.card} onClick={onClick}>
      <div style={styles.header}>
        <div style={styles.avatar}>
          {candidate.handle.charAt(0).toUpperCase()}
        </div>
        <div style={styles.info}>
          <div style={styles.name}>
            {candidate.name || `@${candidate.handle}`}
          </div>
          <div style={styles.handle}>
            @{candidate.handle} · {candidate.followers_count.toLocaleString()} followers
          </div>
        </div>
        <div style={styles.score}>
          <span style={styles.scoreValue}>{candidate.final_score.toFixed(0)}</span>
          <span style={styles.scoreLabel}>Score</span>
        </div>
      </div>

      <div style={styles.bio}>{candidate.bio}</div>

      <div style={styles.meta}>
        <span
          style={{
            ...styles.role,
            backgroundColor: roleColors[candidate.recommended_role] || roleColors.none,
          }}
        >
          {candidate.recommended_role}
        </span>

        {candidate.github_url && (
          <a
            href={candidate.github_url}
            target="_blank"
            rel="noopener noreferrer"
            style={styles.link}
            onClick={(e) => e.stopPropagation()}
          >
            GitHub
          </a>
        )}
        {candidate.linkedin_url && (
          <a
            href={candidate.linkedin_url}
            target="_blank"
            rel="noopener noreferrer"
            style={styles.link}
            onClick={(e) => e.stopPropagation()}
          >
            LinkedIn
          </a>
        )}
      </div>

      {candidate.strengths.length > 0 && (
        <div style={styles.reasoning}>
          <div style={styles.reasoningLabel}>
            <ScoreDot score={candidate.final_score} maxScore={100} />
            Key Strengths
          </div>
          <ul style={styles.list}>
            {candidate.strengths.slice(0, 3).map((s, i) => (
              <li key={i} style={styles.listItem}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={styles.summary}>{candidate.summary}</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '12px',
    padding: '20px',
    marginBottom: '16px',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  header: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '12px',
    marginBottom: '12px',
  },
  avatar: {
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    backgroundColor: 'var(--bg-tertiary)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 600,
    color: 'var(--text-secondary)',
  },
  info: {
    flex: 1,
  },
  name: {
    fontSize: '16px',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  handle: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
  },
  score: {
    textAlign: 'right' as const,
  },
  scoreValue: {
    display: 'block',
    fontSize: '24px',
    fontWeight: 700,
    color: 'var(--accent-green)',
  },
  scoreLabel: {
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  bio: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    marginBottom: '12px',
    lineHeight: 1.5,
  },
  meta: {
    display: 'flex',
    gap: '8px',
    marginBottom: '12px',
    flexWrap: 'wrap' as const,
  },
  role: {
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    fontWeight: 500,
    color: 'white',
    textTransform: 'capitalize' as const,
  },
  link: {
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    color: 'var(--accent-blue)',
    textDecoration: 'none',
    backgroundColor: 'var(--bg-tertiary)',
  },
  reasoning: {
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '8px',
    padding: '12px',
    marginBottom: '12px',
  },
  reasoningLabel: {
    fontSize: '12px',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
  },
  list: {
    margin: 0,
    paddingLeft: '16px',
  },
  listItem: {
    fontSize: '13px',
    color: 'var(--text-primary)',
    marginBottom: '4px',
  },
  summary: {
    fontSize: '13px',
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
  },
};
