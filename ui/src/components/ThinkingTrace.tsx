import React, { useState, useEffect, useRef } from 'react';

interface ThinkingTraceProps {
  thinking: string;
  criteria?: string;
  isStreaming?: boolean;
}

export function ThinkingTrace({ thinking, criteria, isStreaming }: ThinkingTraceProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const contentRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom while streaming
  useEffect(() => {
    if (isStreaming && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [thinking, isStreaming]);

  if (!thinking) return null;

  return (
    <div style={styles.container}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        style={styles.header}
      >
        <span style={styles.icon}>{isExpanded ? '▼' : '▶'}</span>
        <span style={styles.label}>
          {isStreaming ? 'Thinking...' : 'Thinking'}
        </span>
        {criteria && !isStreaming && <span style={styles.criteria}>{criteria}</span>}
      </button>

      {isExpanded && (
        <div style={styles.content} ref={contentRef}>
          <pre style={styles.text}>
            {thinking}
            {isStreaming && <span style={styles.cursor}>▊</span>}
          </pre>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginBottom: '20px',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
    overflow: 'hidden',
  },
  header: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '14px 16px',
    background: 'var(--bg-secondary)',
    border: 'none',
    cursor: 'pointer',
    textAlign: 'left',
  },
  icon: {
    fontSize: '10px',
    color: 'var(--text-muted)',
  },
  label: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  criteria: {
    flex: 1,
    fontSize: '13px',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  content: {
    padding: '16px',
    background: 'var(--bg-tertiary)',
    borderTop: '1px solid var(--border-color)',
    maxHeight: '300px',
    overflowY: 'auto' as const,
  },
  text: {
    margin: 0,
    fontSize: '13px',
    lineHeight: 1.6,
    color: 'var(--text-secondary)',
    whiteSpace: 'pre-wrap' as const,
    fontFamily: 'inherit',
  },
  cursor: {
    color: 'var(--accent-primary)',
    animation: 'blink 1s infinite',
  },
};
