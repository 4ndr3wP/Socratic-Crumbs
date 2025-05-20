import React, { useState, useEffect } from 'react';
import useTTSStreaming from '../hooks/useTTSStreaming';

function TTSButton({ isStreaming, currentText, onTTSToggle }) {
  const [isTTSEnabled, setIsTTSEnabled] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const { startStreaming, stopStreaming, status, error } = useTTSStreaming();

  // Debug logging
  useEffect(() => {
    console.log('TTSButton props:', { isStreaming, currentText, ttsStatus: status });
  }, [isStreaming, currentText, status]);

  // Handle status changes from the streaming hook
  useEffect(() => {
    if (status === 'error' && error) {
      console.error('TTS streaming error:', error);
      setIsPlaying(false);
    }
    else if (status === 'streaming') {
      setIsPlaying(true);
    }
    else if (status === 'disconnected' || status === 'idle') {
      setIsPlaying(false);
    }
  }, [status, error]);

  // Handle new text to speak when TTS is enabled
  useEffect(() => {
    if (isTTSEnabled && currentText && !isPlaying) {
      // Start streaming with a small delay to avoid multiple rapid requests
      const timeoutId = setTimeout(() => {
        console.log('Starting TTS streaming for text:', currentText.substring(0, 50) + '...');
        startStreaming(currentText);
      }, 300);
      
      return () => clearTimeout(timeoutId);
    }
  }, [isTTSEnabled, currentText, isPlaying, startStreaming]);

  const handleTTS = () => {
    const newState = !isTTSEnabled;
    setIsTTSEnabled(newState);
    
    if (!newState && isPlaying) {
      // If turning off TTS while playing, stop current playback
      stopStreaming();
      setIsPlaying(false);
    }
    
    onTTSToggle(newState);
  };

  return (
    <button
      className={`tts-button ${isTTSEnabled ? 'enabled' : ''} ${isPlaying ? 'playing' : ''}`}
      onClick={handleTTS}
      title={isTTSEnabled ? "Disable TTS" : "Enable TTS"}
      style={{ 
        cursor: 'pointer',
        animation: isPlaying ? 'pulse 1.5s infinite' : 'none',
        marginRight: '8px', // Match mic icon spacing
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g>
          <polygon points="6,16 20,16 28,8 28,40 20,32 6,32" fill="#fff" stroke={isTTSEnabled || isPlaying ? '#7C3AED' : '#333'} strokeWidth="2.5"/>
          <path d="M34 18C36.6667 20.6667 36.6667 27.3333 34 30" stroke={isTTSEnabled || isPlaying ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
          <path d="M38 14C42.6667 18.6667 42.6667 29.3333 38 34" stroke={isTTSEnabled || isPlaying ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
          <path d="M42 10C49.3333 17.3333 49.3333 30.6667 42 38" stroke={isTTSEnabled || isPlaying ? '#7C3AED' : '#333'} strokeWidth="2.5" fill="none"/>
        </g>
      </svg>
    </button>
  );
}

export default TTSButton; 