import { useEffect } from 'react';
import { handleAuthCallback } from '../lib/auth';

interface AuthCallbackProps {
  onAuthSuccess: () => void;
  onAuthError: (error: string) => void;
}

export function AuthCallback({ onAuthSuccess, onAuthError }: AuthCallbackProps) {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const error = params.get('error');

    if (error) {
      onAuthError(error);
      // Redirect to home
      window.location.href = '/';
      return;
    }

    const authState = handleAuthCallback();

    if (authState) {
      onAuthSuccess();
      // Redirect to home
      window.location.href = '/';
    } else {
      onAuthError('Failed to authenticate');
      window.location.href = '/';
    }
  }, [onAuthSuccess, onAuthError]);

  return (
    <div style={styles.container}>
      <div style={styles.spinner} />
      <p style={styles.text}>Completing login...</p>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    gap: '16px',
  },
  spinner: {
    width: '32px',
    height: '32px',
    border: '3px solid var(--border-color)',
    borderTopColor: 'var(--accent-primary)',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  text: {
    fontSize: '16px',
    color: 'var(--text-secondary)',
  },
};
