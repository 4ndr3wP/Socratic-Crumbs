import React, { useState } from 'react';

const TTSButton = ({ isStreaming, currentText, onTTSToggle }) => {
  const [isEnabled, setIsEnabled] = useState(false);

  const handleClick = () => {
    const newState = !isEnabled;
    setIsEnabled(newState);
    onTTSToggle(newState);
  };

  return (
    <button
      className={`tts-button${isEnabled ? ' streaming' : ''}`}
      onClick={handleClick}
      title={isEnabled ? "Stop TTS" : "Start TTS"}
      style={{
        background: 'none',
        border: 'none',
        padding: '0',
        cursor: 'pointer',
        opacity: isEnabled ? 1 : 0.7,
        transition: 'opacity 0.2s',
        marginLeft: '0',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
      }}
    >
      {isEnabled && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          animation: 'pulse 2s infinite',
        }} />
      )}
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ position: 'relative', zIndex: 1 }}>
        <path d="M4 9V15H8L14 21V3L8 9H4Z" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M19 5C20.3333 6.33333 21 8 21 12C21 16 20.3333 17.6667 19 19" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M17 8C17.6667 9.33333 18 10.6667 18 12C18 13.3333 17.6667 14.6667 17 16" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      <style>
        {`
          @keyframes pulse {
            0% {
              transform: translate(-50%, -50%) scale(0.8);
              opacity: 0.8;
            }
            50% {
              transform: translate(-50%, -50%) scale(1.2);
              opacity: 0.4;
            }
            100% {
              transform: translate(-50%, -50%) scale(0.8);
              opacity: 0.8;
            }
          }
        `}
      </style>
    </button>
  );
};

export default TTSButton; 