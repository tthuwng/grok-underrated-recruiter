import React from 'react';

interface HeartButtonProps {
  isSaved: boolean;
  onToggle: () => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export function HeartButton({
  isSaved,
  onToggle,
  disabled = false,
  size = 'medium',
}: HeartButtonProps) {
  const sizeMap = {
    small: '18px',
    medium: '22px',
    large: '28px',
  };

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        if (!disabled) onToggle();
      }}
      style={{
        ...styles.button,
        fontSize: sizeMap[size],
        color: isSaved ? 'var(--accent-red)' : 'var(--text-muted)',
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      disabled={disabled}
      title={isSaved ? 'Remove from saved' : 'Save candidate'}
    >
      {isSaved ? '♥' : '♡'}
    </button>
  );
}

const styles: Record<string, React.CSSProperties> = {
  button: {
    background: 'none',
    border: 'none',
    padding: '4px 8px',
    lineHeight: 1,
    transition: 'transform 0.15s ease, color 0.15s ease',
  },
};
