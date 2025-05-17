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
        animation: isPlaying ? 'pulse 1.5s infinite' : 'none'
      }}
    >
      {isPlaying ? "🔊" : isTTSEnabled ? "🔉" : "🔈"}
    </button>
  );
}

export default TTSButton; 