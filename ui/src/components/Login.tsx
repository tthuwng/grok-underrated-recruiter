import React from 'react';
import { loginWithX, logout, type User } from '../lib/auth';

interface LoginProps {
  user: User | null;
  onLogout: () => void;
}

export function Login({ user, onLogout }: LoginProps) {
  const handleLogin = () => {
    loginWithX();
  };

  const handleLogout = async () => {
    await logout();
    onLogout();
  };

  if (user) {
    return (
      <div style={styles.container}>
        <div style={styles.userInfo}>
          <span style={styles.handle}>@{user.x_handle}</span>
        </div>
        <button style={styles.logoutButton} onClick={handleLogout}>
          Logout
        </button>
      </div>
    );
  }

  return (
    <button style={styles.loginButton} onClick={handleLogin}>
      <XLogo />
      <span>Login with X</span>
    </button>
  );
}

function XLogo() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="currentColor"
      style={{ marginRight: '8px' }}
    >
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  handle: {
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--text-primary)',
  },
  loginButton: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
    fontSize: '14px',
    fontWeight: 500,
    color: 'white',
    backgroundColor: '#000',
    border: '1px solid #333',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  logoutButton: {
    padding: '6px 12px',
    fontSize: '13px',
    color: 'var(--text-secondary)',
    backgroundColor: 'transparent',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    cursor: 'pointer',
  },
};
