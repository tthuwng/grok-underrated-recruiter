import React, { useState } from 'react';
import { submitHandle } from '../api/client';
import { getAuthToken } from '../lib/auth';

interface SubmitHandleFormProps {
  onSubmitSuccess: (submissionId: string) => void;
}

export function SubmitHandleForm({ onSubmitSuccess }: SubmitHandleFormProps) {
  const [handle, setHandle] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const cleanHandle = handle.trim().replace(/^@/, '');
    if (!cleanHandle) {
      setError('Please enter a handle');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const token = getAuthToken();
      const result = await submitHandle(cleanHandle, token || undefined);
      setSuccess(`Submitted @${cleanHandle} for review`);
      setHandle('');
      onSubmitSuccess(result.submission_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit handle');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.inputGroup}>
        <span style={styles.atSymbol}>@</span>
        <input
          type="text"
          value={handle}
          onChange={(e) => setHandle(e.target.value)}
          placeholder="username"
          style={styles.input}
          disabled={isLoading}
        />
      </div>

      <button type="submit" style={styles.submitButton} disabled={isLoading}>
        {isLoading ? 'Submitting...' : 'Submit for Review'}
      </button>

      {error && <div style={styles.error}>{error}</div>}
      {success && <div style={styles.success}>{success}</div>}

      <p style={styles.hint}>
        Submissions require admin approval before processing.
        Limited to 5 submissions per hour.
      </p>
    </form>
  );
}

const styles: Record<string, React.CSSProperties> = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  inputGroup: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '8px',
    border: '1px solid var(--border-color)',
    overflow: 'hidden',
  },
  atSymbol: {
    padding: '12px 0 12px 16px',
    fontSize: '15px',
    color: 'var(--text-muted)',
    fontWeight: 500,
  },
  input: {
    flex: 1,
    padding: '12px 16px 12px 4px',
    fontSize: '15px',
    backgroundColor: 'transparent',
    border: 'none',
    color: 'var(--text-primary)',
    outline: 'none',
  },
  submitButton: {
    padding: '12px 20px',
    fontSize: '14px',
    fontWeight: 600,
    color: '#000',
    backgroundColor: 'var(--accent-primary)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
  error: {
    padding: '10px 14px',
    fontSize: '13px',
    color: 'var(--accent-red)',
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    borderRadius: '8px',
  },
  success: {
    padding: '10px 14px',
    fontSize: '13px',
    color: 'var(--accent-green)',
    backgroundColor: 'rgba(52, 211, 153, 0.1)',
    borderRadius: '8px',
  },
  hint: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    margin: 0,
    lineHeight: 1.5,
  },
};
