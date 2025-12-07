import React, { useState } from 'react';
import type { ScoreBreakdown as ScoreBreakdownType } from '../api/client';

interface ScoreBarProps {
  label: string;
  score: number;
  evidence: string;
  weight: string;
}

function ScoreBar({ label, score, evidence, weight }: ScoreBarProps) {
  const percentage = (score / 10) * 100;

  // Color based on score
  const getColor = (s: number) => {
    if (s >= 8) return 'var(--accent-green)';
    if (s >= 6) return 'var(--accent-blue)';
    if (s >= 4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  return (
    <div style={styles.scoreBar}>
      <div style={styles.scoreHeader}>
        <span style={styles.scoreLabel}>{label}</span>
        <span style={styles.scoreWeight}>{weight}</span>
        <span style={styles.scoreValue}>{score}/10</span>
      </div>
      <div style={styles.barContainer}>
        <div
          style={{
            ...styles.barFill,
            width: `${percentage}%`,
            backgroundColor: getColor(score),
          }}
        />
      </div>
      {evidence && (
        <p style={styles.evidence}>{evidence}</p>
      )}
    </div>
  );
}

interface ScoreBreakdownProps {
  technicalDepth?: ScoreBreakdownType;
  projectEvidence?: ScoreBreakdownType;
  missionAlignment?: ScoreBreakdownType;
  exceptionalAbility?: ScoreBreakdownType;
  communication?: ScoreBreakdownType;
  defaultExpanded?: boolean;
}

export function ScoreBreakdown({
  technicalDepth,
  projectEvidence,
  missionAlignment,
  exceptionalAbility,
  communication,
  defaultExpanded = false,
}: ScoreBreakdownProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  // Check if any scores exist
  const hasScores = technicalDepth || projectEvidence || missionAlignment ||
                    exceptionalAbility || communication;

  if (!hasScores) {
    return null;
  }

  return (
    <div style={styles.container}>
      <button
        style={styles.toggleButton}
        onClick={(e) => {
          e.stopPropagation();
          setIsExpanded(!isExpanded);
        }}
      >
        <span style={styles.toggleIcon}>{isExpanded ? '▼' : '▶'}</span>
        <span>Score Breakdown</span>
      </button>

      {isExpanded && (
        <div style={styles.content}>
          {technicalDepth && (
            <ScoreBar
              label="Technical Depth"
              score={technicalDepth.score}
              evidence={technicalDepth.evidence}
              weight="25%"
            />
          )}
          {projectEvidence && (
            <ScoreBar
              label="Project Evidence"
              score={projectEvidence.score}
              evidence={projectEvidence.evidence}
              weight="25%"
            />
          )}
          {missionAlignment && (
            <ScoreBar
              label="Mission Alignment"
              score={missionAlignment.score}
              evidence={missionAlignment.evidence}
              weight="20%"
            />
          )}
          {exceptionalAbility && (
            <ScoreBar
              label="Exceptional Ability"
              score={exceptionalAbility.score}
              evidence={exceptionalAbility.evidence}
              weight="20%"
            />
          )}
          {communication && (
            <ScoreBar
              label="Communication"
              score={communication.score}
              evidence={communication.evidence}
              weight="10%"
            />
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: '12px',
    borderTop: '1px solid var(--border-color)',
    paddingTop: '12px',
  },
  toggleButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    background: 'none',
    border: 'none',
    color: 'var(--text-secondary)',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    padding: '4px 0',
  },
  toggleIcon: {
    fontSize: '10px',
  },
  content: {
    marginTop: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  scoreBar: {},
  scoreHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
  },
  scoreLabel: {
    fontSize: '13px',
    fontWeight: 500,
    color: 'var(--text-primary)',
    flex: 1,
  },
  scoreWeight: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    backgroundColor: 'var(--bg-secondary)',
    padding: '2px 6px',
    borderRadius: '4px',
  },
  scoreValue: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    minWidth: '40px',
    textAlign: 'right',
  },
  barContainer: {
    height: '6px',
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  },
  evidence: {
    margin: '6px 0 0 0',
    fontSize: '12px',
    color: 'var(--text-muted)',
    lineHeight: 1.4,
    fontStyle: 'italic',
  },
};
