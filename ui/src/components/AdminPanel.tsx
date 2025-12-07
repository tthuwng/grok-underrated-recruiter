import React, { useState, useEffect, useCallback } from 'react';
import { getPendingApprovals, approveSubmission, rejectSubmission } from '../api/client';
import type { PendingApproval } from '../api/client';
import { getAuthToken, type User } from '../lib/auth';

interface AdminPanelProps {
  user: User;
  onBack: () => void;
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

export function AdminPanel({ user, onBack }: AdminPanelProps) {
  const [pending, setPending] = useState<PendingApproval[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  const loadPending = useCallback(async () => {
    const token = getAuthToken();
    if (!token) {
      setError('Not authenticated');
      setIsLoading(false);
      return;
    }

    try {
      const results = await getPendingApprovals(token);
      setPending(results);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load pending approvals');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPending();
  }, [loadPending]);

  // Auto-refresh every 10 seconds
  useEffect(() => {
    const interval = setInterval(loadPending, 10000);
    return () => clearInterval(interval);
  }, [loadPending]);

  const handleApprove = async (submissionId: string) => {
    const token = getAuthToken();
    if (!token) return;

    setActionInProgress(submissionId);
    try {
      await approveSubmission(submissionId, token);
      setPending((prev) => prev.filter((p) => p.submission_id !== submissionId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to approve');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleReject = async (submissionId: string) => {
    const token = getAuthToken();
    if (!token) return;

    setActionInProgress(submissionId);
    try {
      await rejectSubmission(submissionId, token);
      setPending((prev) => prev.filter((p) => p.submission_id !== submissionId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reject');
    } finally {
      setActionInProgress(null);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button onClick={onBack} style={styles.backButton}>
          &larr; Back to Search
        </button>
        <div style={styles.titleRow}>
          <h1 style={styles.title}>Admin Panel</h1>
          <span style={styles.adminBadge}>Admin: @{user.x_handle}</span>
        </div>
      </div>

      <div style={styles.content}>
        <div style={styles.section}>
          <div style={styles.sectionHeader}>
            <h2 style={styles.sectionTitle}>
              Pending Approvals {pending.length > 0 && `(${pending.length})`}
            </h2>
            <button onClick={loadPending} style={styles.refreshButton} disabled={isLoading}>
              {isLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>

          {error && <div style={styles.error}>{error}</div>}

          {isLoading && pending.length === 0 ? (
            <div style={styles.loading}>Loading pending approvals...</div>
          ) : pending.length === 0 ? (
            <div style={styles.empty}>
              No pending approvals. All caught up!
            </div>
          ) : (
            <div style={styles.list}>
              {pending.map((item) => (
                <div key={item.submission_id} style={styles.card}>
                  <div style={styles.cardHeader}>
                    <div style={styles.handleSection}>
                      <span style={styles.handle}>@{item.handle}</span>
                      <span style={styles.submittedBy}>
                        {item.submitted_by
                          ? `Submitted by @${item.submitted_by}`
                          : 'Anonymous submission'}
                        {' · '}
                        {formatTimeAgo(item.submitted_at)}
                      </span>
                    </div>
                    <div style={styles.statusBadge}>Pending</div>
                  </div>

                  <div style={styles.actions}>
                    <button
                      onClick={() => handleApprove(item.submission_id)}
                      style={styles.approveButton}
                      disabled={actionInProgress === item.submission_id}
                    >
                      {actionInProgress === item.submission_id ? '...' : '✓ Approve'}
                    </button>
                    <button
                      onClick={() => handleReject(item.submission_id)}
                      style={styles.rejectButton}
                      disabled={actionInProgress === item.submission_id}
                    >
                      {actionInProgress === item.submission_id ? '...' : '✗ Reject'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    flex: 1,
    maxWidth: '700px',
    width: '100%',
    margin: '0 auto',
    padding: '24px',
  },
  header: {
    marginBottom: '24px',
  },
  backButton: {
    padding: '8px 0',
    fontSize: '14px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: 'none',
    cursor: 'pointer',
    marginBottom: '8px',
  },
  titleRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap',
  },
  title: {
    fontSize: '24px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    margin: 0,
  },
  adminBadge: {
    padding: '4px 10px',
    fontSize: '12px',
    fontWeight: 500,
    color: 'var(--accent-orange)',
    backgroundColor: 'rgba(251, 146, 60, 0.15)',
    borderRadius: '6px',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '32px',
  },
  section: {},
  sectionHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: 600,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    margin: 0,
  },
  refreshButton: {
    padding: '6px 12px',
    fontSize: '12px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  loading: {
    padding: '24px',
    textAlign: 'center',
    color: 'var(--text-muted)',
    fontSize: '14px',
  },
  empty: {
    padding: '24px',
    textAlign: 'center',
    color: 'var(--text-muted)',
    fontSize: '14px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
  },
  error: {
    padding: '10px 14px',
    fontSize: '13px',
    color: 'var(--accent-red)',
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    borderRadius: '8px',
    marginBottom: '16px',
  },
  list: {},
  card: {
    padding: '16px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
    marginBottom: '12px',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '12px',
    marginBottom: '12px',
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
  submittedBy: {
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  statusBadge: {
    padding: '4px 10px',
    fontSize: '12px',
    fontWeight: 500,
    color: 'var(--accent-orange)',
    backgroundColor: 'rgba(251, 146, 60, 0.15)',
    borderRadius: '6px',
    whiteSpace: 'nowrap',
  },
  actions: {
    display: 'flex',
    gap: '8px',
  },
  approveButton: {
    flex: 1,
    padding: '10px 16px',
    fontSize: '13px',
    fontWeight: 600,
    color: '#fff',
    backgroundColor: 'var(--accent-green)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
  rejectButton: {
    flex: 1,
    padding: '10px 16px',
    fontSize: '13px',
    fontWeight: 600,
    color: '#fff',
    backgroundColor: 'var(--accent-red)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
};
