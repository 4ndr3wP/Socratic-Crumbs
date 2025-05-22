import React, { useState, useEffect, useCallback } from 'react';

const STSButton = ({ selectedModel, selectedVoice }) => {
  const [isSTSActive, setIsSTSActive] = useState(false);
  const [status, setStatus] = useState('idle');
  const [websocket, setWebsocket] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);

  // Create a WebSocket connection when the STS is activated
  const startSTS = useCallback(() => {
    if (isSTSActive) return;
    
    // Generate a unique session ID
    const newSessionId = Math.random().toString(36).substring(2, 15);
    setSessionId(newSessionId);
    
    // Create WebSocket connection with the selected model and voice
    const wsUrl = `ws://${window.location.host}/api/sts/${newSessionId}?model=${encodeURIComponent(selectedModel)}&voice=${encodeURIComponent(selectedVoice)}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('STS WebSocket connection established');
      setStatus('connecting');
      setMessages([{ type: 'system', text: 'STS session starting...' }]);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('STS message received:', data);
        
        // Update status based on the message
        setStatus(data.status || 'unknown');
        
        // Add message to the log
        setMessages(prev => [...prev, { 
          type: data.status === 'error' ? 'error' : 'info', 
          text: data.message 
        }]);
        
        // Handle specific status messages
        if (data.status === 'ready') {
          setIsSTSActive(true);
        }
      } catch (error) {
        console.error('Error parsing STS message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('STS WebSocket error:', error);
      setStatus('error');
      setMessages(prev => [...prev, { 
        type: 'error', 
        text: 'Connection error. Please try again.' 
      }]);
    };
    
    ws.onclose = () => {
      console.log('STS WebSocket connection closed');
      setStatus('idle');
      setIsSTSActive(false);
      setMessages(prev => [...prev, { 
        type: 'system', 
        text: 'STS session ended' 
      }]);
    };
    
    setWebsocket(ws);
  }, [isSTSActive]);
  
  // Stop the STS session
  const stopSTS = useCallback(() => {
    if (!websocket) return;
    
    try {
      // Send stop command
      websocket.send('stop');
      
      // Close the connection
      websocket.close();
    } catch (error) {
      console.error('Error stopping STS:', error);
    }
    
    setWebsocket(null);
    setIsSTSActive(false);
    setStatus('idle');
  }, [websocket]);
  
  // Handle button click
  const handleSTSToggle = () => {
    if (isSTSActive) {
      stopSTS();
    } else {
      startSTS();
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, [websocket]);
  
  // Status indicator color
  const getStatusColor = () => {
    switch (status) {
      case 'listening':
      case 'speech_detected':
        return '#38bdf8'; // Blue for listening
      case 'processing':
      case 'transcribed':
      case 'generating':
        return '#a855f7'; // Purple for processing
      case 'speaking':
        return '#22c55e'; // Green for speaking
      case 'error':
        return '#ef4444'; // Red for error
      case 'ready':
        return '#22c55e'; // Green for ready
      default:
        return '#9ca3af'; // Gray for idle/connecting
    }
  };
  
  // Get button text based on status
  const getButtonText = () => {
    if (!isSTSActive) return 'Start Voice Chat';
    
    switch (status) {
      case 'listening':
        return 'Listening...';
      case 'speech_detected':
        return 'Hearing you...';
      case 'processing':
        return 'Processing...';
      case 'generating':
        return 'Thinking...';
      case 'speaking':
        return 'Speaking...';
      default:
        return 'Stop Voice Chat';
    }
  };
  
  return (
    <div className="sts-container">
      <button
        className={`sts-button ${isSTSActive ? 'active' : ''}`}
        onClick={handleSTSToggle}
        style={{
          backgroundColor: isSTSActive ? getStatusColor() : '#f3f4f6',
          color: isSTSActive ? 'white' : '#4b5563',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 16px',
          fontSize: '14px',
          fontWeight: 'bold',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s ease',
          boxShadow: isSTSActive ? '0 2px 5px rgba(0,0,0,0.2)' : 'none',
          marginBottom: '10px',
          width: '100%'
        }}
      >
        {isSTSActive && (
          <span 
            className="pulse-indicator"
            style={{
              display: 'inline-block',
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: 'white',
              marginRight: '8px',
              animation: isSTSActive ? 'pulse 1.5s infinite' : 'none'
            }}
          />
        )}
        {getButtonText()}
      </button>
      
      {/* Status messages - only show when active */}
      {isSTSActive && messages.length > 0 && (
        <div 
          className="sts-messages"
          style={{
            maxHeight: '100px',
            overflowY: 'auto',
            fontSize: '12px',
            marginTop: '5px',
            padding: '5px',
            borderRadius: '4px',
            backgroundColor: '#f9fafb',
            border: '1px solid #e5e7eb'
          }}
        >
          {messages.slice(-3).map((msg, idx) => (
            <div 
              key={idx} 
              style={{
                color: msg.type === 'error' ? '#ef4444' : 
                      msg.type === 'system' ? '#9ca3af' : '#4b5563',
                marginBottom: '2px'
              }}
            >
              {msg.text}
            </div>
          ))}
        </div>
      )}
      
      {/* CSS for the pulse animation */}
      <style>
        {`
          @keyframes pulse {
            0% {
              transform: scale(0.95);
              box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7);
            }
            
            70% {
              transform: scale(1);
              box-shadow: 0 0 0 5px rgba(255, 255, 255, 0);
            }
            
            100% {
              transform: scale(0.95);
              box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
            }
          }
        `}
      </style>
    </div>
  );
};

export default STSButton;
