import React from 'react';

function StopButton({ onStop, isStreaming, isAudioPlaying }) {
  const handleClick = () => {
    // Stop both text streaming and audio playback
    onStop();
  };

  const isActive = isStreaming || isAudioPlaying;

  return (
    <button
      onClick={handleClick}
      disabled={!isActive}
      style={{
        padding: '8px 16px',
        backgroundColor: isActive ? '#ef4444' : '#f3f4f6',
        color: isActive ? 'white' : '#9ca3af',
        border: 'none',
        borderRadius: '6px',
        cursor: isActive ? 'pointer' : 'not-allowed',
        fontSize: '14px',
        fontWeight: '500',
        transition: 'all 0.2s ease',
        display: 'flex',
        alignItems: 'center',
        gap: '6px'
      }}
      title={isActive ? "Stop generation and audio" : "No active generation or audio"}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect x="6" y="6" width="12" height="12" />
      </svg>
      Stop
    </button>
  );
}

export default StopButton; 