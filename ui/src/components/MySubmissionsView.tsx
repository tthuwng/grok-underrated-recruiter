import React, { useState, useEffect, useCallback } from 'react';
import { SubmitHandleForm } from './SubmitHandleForm';
import { SubmissionStatusCard } from './SubmissionStatusCard';
import { getSubmissionStatus } from '../api/client';
import type { SubmissionStatus } from '../api/client';
import {
  getStoredSubmissions,
  addStoredSubmission,
} from '../lib/auth';

interface MySubmissionsViewProps {
  onBack: () => void;
}

export function MySubmissionsView({ onBack }: MySubmissionsViewProps) {
  const [submissions, setSubmissions] = useState<SubmissionStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadSubmissions = useCallback(async () => {
    const storedIds = getStoredSubmissions();
    if (storedIds.length === 0) {
      setSubmissions([]);
      setIsLoading(false);
      return;
    }

    try {
      const results = await Promise.all(
        storedIds.map(async (id) => {
          try {
            return await getSubmissionStatus(id);
          } catch {
            return null; // Submission may have been deleted
          }
        })
      );

      setSubmissions(results.filter((s): s is SubmissionStatus => s !== null));
    } catch (e) {
      console.error('Failed to load submissions:', e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSubmissions();
  }, [loadSubmissions]);

  // Auto-refresh submissions that are processing
  useEffect(() => {
    const hasProcessing = submissions.some(
      (s) => s.status === 'processing' || (s.status === 'pending' && s.approval_status === 'approved')
    );

    if (!hasProcessing) return;

    const interval = setInterval(loadSubmissions, 5000);
    return () => clearInterval(interval);
  }, [submissions, loadSubmissions]);

  const handleSubmitSuccess = (submissionId: string) => {
    addStoredSubmission(submissionId);
    loadSubmissions();
  };

  const handleRefresh = async (submissionId: string) => {
    try {
      const updated = await getSubmissionStatus(submissionId);
      setSubmissions((prev) =>
        prev.map((s) => (s.submission_id === submissionId ? updated : s))
      );
    } catch (e) {
      console.error('Failed to refresh submission:', e);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button onClick={onBack} style={styles.backButton}>
          &larr; Back to Search
        </button>
        <h1 style={styles.title}>Submit Handle</h1>
      </div>

      <div style={styles.content}>
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Submit an X Handle for Evaluation</h2>
          <SubmitHandleForm onSubmitSuccess={handleSubmitSuccess} />
        </div>

        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>
            Your Submissions {submissions.length > 0 && `(${submissions.length})`}
          </h2>

          {isLoading ? (
            <div style={styles.loading}>Loading submissions...</div>
          ) : submissions.length === 0 ? (
            <div style={styles.empty}>
              No submissions yet. Submit an X handle above to get started.
            </div>
          ) : (
            <div style={styles.list}>
              {submissions.map((submission) => (
                <SubmissionStatusCard
                  key={submission.submission_id}
                  submission={submission}
                  onRefresh={() => handleRefresh(submission.submission_id)}
                />
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
    maxWidth: '600px',
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
  title: {
    fontSize: '24px',
    fontWeight: 600,
    color: 'var(--text-primary)',
    margin: 0,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '32px',
  },
  section: {},
  sectionTitle: {
    fontSize: '14px',
    fontWeight: 600,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '16px',
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
  list: {},
};
