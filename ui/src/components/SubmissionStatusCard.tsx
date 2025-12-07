import React from 'react';
import type { SubmissionStatus } from '../api/client';

interface SubmissionStatusCardProps {
  submission: SubmissionStatus;
  onRefresh?: () => void;
}

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
  return `${Math.floor(seconds / 86400)} days ago`;
}

function getStatusColor(status: string, approvalStatus: string): string {
  if (approvalStatus === 'rejected') return 'var(--accent-red)';
  if (approvalStatus === 'pending_approval') return 'var(--accent-orange)';

  switch (status) {
    case 'completed': return 'var(--accent-green)';
    case 'processing': return 'var(--accent-blue)';
    case 'failed': return 'var(--accent-red)';
    case 'filtered_out': return 'var(--text-muted)';
    default: return 'var(--accent-blue)';
  }
}

function getStatusLabel(status: string, approvalStatus: string, stage: string): string {
  if (approvalStatus === 'rejected') return 'Rejected';
  if (approvalStatus === 'pending_approval') return 'Pending Approval';

  switch (status) {
    case 'completed': return 'Completed';
    case 'processing': return `Processing (${stage})`;
    case 'failed': return 'Failed';
    case 'filtered_out': return 'Filtered Out';
    case 'pending': return 'In Queue';
    default: return status;
  }
}

export function SubmissionStatusCard({ submission, onRefresh }: SubmissionStatusCardProps) {
  const statusColor = getStatusColor(submission.status, submission.approval_status);
  const statusLabel = getStatusLabel(submission.status, submission.approval_status, submission.stage);
  const isProcessing = submission.status === 'processing';

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.handleSection}>
          <span style={styles.handle}>@{submission.handle}</span>
          <span style={styles.time}>{formatTimeAgo(submission.submitted_at)}</span>
        </div>
        <div style={{ ...styles.statusBadge, backgroundColor: `${statusColor}20`, color: statusColor }}>
          {isProcessing && <span style={styles.pulse} />}
          {statusLabel}
        </div>
      </div>

      {submission.position_in_queue && submission.approval_status === 'approved' && submission.status === 'pending' && (
        <div style={styles.queuePosition}>
          Position in queue: <strong>{submission.position_in_queue}</strong>
        </div>
      )}

      {submission.error && (
        <div style={styles.error}>
          Error: {submission.error}
        </div>
      )}

      {submission.approved_by && (
        <div style={styles.meta}>
          {submission.approval_status === 'approved' ? 'Approved' : 'Rejected'} by @{submission.approved_by}
        </div>
      )}

      {onRefresh && (isProcessing || submission.status === 'pending') && (
        <button style={styles.refreshButton} onClick={onRefresh}>
          Refresh
        </button>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    padding: '16px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
    marginBottom: '12px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '12px',
  },
  handleSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  handle: {
    fontSize: '15px',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  time: {
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  statusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    fontSize: '12px',
    fontWeight: 500,
    borderRadius: '6px',
    whiteSpace: 'nowrap',
  },
  pulse: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: 'currentColor',
    animation: 'pulse 1.5s infinite',
  },
  queuePosition: {
    marginTop: '12px',
    fontSize: '13px',
    color: 'var(--text-secondary)',
  },
  error: {
    marginTop: '12px',
    padding: '8px 12px',
    fontSize: '13px',
    color: 'var(--accent-red)',
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    borderRadius: '6px',
  },
  meta: {
    marginTop: '12px',
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  refreshButton: {
    marginTop: '12px',
    padding: '6px 12px',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    cursor: 'pointer',
  },
};
