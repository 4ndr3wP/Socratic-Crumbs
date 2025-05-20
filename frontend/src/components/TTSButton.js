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
      <svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g>
          <polygon points="6,16 20,16 28,8 28,40 20,32 6,32" fill="#fff" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2.5"/>
          <path d="M34 18C36.6667 20.6667 36.6667 27.3333 34 30" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
          <path d="M38 14C42.6667 18.6667 42.6667 29.3333 38 34" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
          <path d="M42 10C49.3333 17.3333 49.3333 30.6667 42 38" stroke={isEnabled ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
        </g>
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