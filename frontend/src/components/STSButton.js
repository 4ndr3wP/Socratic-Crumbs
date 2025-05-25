import React, { useState, useEffect, useCallback } from 'react';

const MIC_ICON = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ marginRight: 8 }}>
    <rect width="20" height="20" fill="none"/>
    <path d="M10 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm5-3a1 1 0 1 0-2 0 3 3 0 0 1-6 0 1 1 0 1 0-2 0 5 5 0 0 0 4 4.9V18a1 1 0 1 0 2 0v-2.1A5 5 0 0 0 15 11Z" fill="currentColor"/>
  </svg>
);

const STSButton = ({ selectedModel, selectedVoice, onSTSActiveChange, onStatusChange, onMessagesChange, isActive }) => {
  const [isSTSActive, setIsSTSActive] = useState(false);
  const [status, setStatus] = useState('idle');
  const [websocket, setWebsocket] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);

  // Sync with external isActive prop
  useEffect(() => {
    setIsSTSActive(isActive);
  }, [isActive]);

  // Notify parent of state changes
  useEffect(() => {
    if (onSTSActiveChange) onSTSActiveChange(isSTSActive);
  }, [isSTSActive, onSTSActiveChange]);
  useEffect(() => {
    if (onStatusChange) onStatusChange(status);
  }, [status, onStatusChange]);
  useEffect(() => {
    if (onMessagesChange) onMessagesChange(messages);
  }, [messages, onMessagesChange]);

  // Create a WebSocket connection when the STS is activated
  const startSTS = useCallback(() => {
    if (isSTSActive) return;
    const newSessionId = Math.random().toString(36).substring(2, 15);
    setSessionId(newSessionId);
    const wsUrl = `ws://${window.location.host}/api/sts/${newSessionId}?model=${encodeURIComponent(selectedModel)}&voice=${encodeURIComponent(selectedVoice)}`;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      setStatus('connecting');
      setMessages([{ type: 'system', text: 'STS session starting...' }]);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.status || 'unknown');
        setMessages(prev => [...prev, { type: data.status === 'error' ? 'error' : 'info', text: data.message }]);
        if (data.status === 'ready') setIsSTSActive(true);
      } catch (error) {
        // ignore
      }
    };
    ws.onerror = () => {
      setStatus('error');
      setMessages(prev => [...prev, { type: 'error', text: 'Connection error. Please try again.' }]);
    };
    ws.onclose = () => {
      setStatus('idle');
      setIsSTSActive(false);
      setMessages(prev => [...prev, { type: 'system', text: 'STS session ended' }]);
    };
    setWebsocket(ws);
  }, [isSTSActive, selectedModel, selectedVoice]);

  // Stop the STS session
  const stopSTS = useCallback(() => {
    if (!websocket) return;
    try {
      // Send stop command and close immediately
      websocket.send('stop');
      // Close the websocket immediately to ensure cleanup
      websocket.close(1000, 'User stopped session');
    } catch (error) {
      console.warn('Error stopping STS session:', error);
    }
    setWebsocket(null);
    setIsSTSActive(false);
    setStatus('idle');
    setMessages(prev => [...prev, { type: 'system', text: 'Voice Chat stopped' }]);
    if (onSTSActiveChange) onSTSActiveChange(false);
  }, [websocket, onSTSActiveChange]);

  // Start or stop STS
  const handleSTSToggle = () => {
    if (isSTSActive) {
      stopSTS();
    } else {
      startSTS();
    }
  };

  useEffect(() => {
    // Cleanup function to ensure session is stopped when component unmounts
    return () => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        try {
          websocket.send('stop');
          websocket.close(1000, 'Component unmounting');
        } catch (error) {
          console.warn('Error cleaning up STS session on unmount:', error);
        }
      }
    };
  }, [websocket]);

  // Button color: always purple when active, grey when inactive
  const buttonColor = isSTSActive ? '#7c3aed' : '#e5e7eb';
  const buttonTextColor = isSTSActive ? 'white' : '#333';

  return (
    <div className="sts-container">
      <button
        className={`sts-button${isSTSActive ? ' active' : ''}`}
        onClick={handleSTSToggle}
        type="button"
        style={{
          backgroundColor: buttonColor,
          color: buttonTextColor,
          border: 'none',
          borderRadius: '18px',
          height: '36px',
          padding: '0 24px',
          fontSize: '15px',
          fontWeight: 'bold',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s ease',
          marginBottom: '10px',
          minWidth: 0,
          width: 'auto',
        }}
      >
        Voice Chat
      </button>
    </div>
  );
};

export default STSButton;
