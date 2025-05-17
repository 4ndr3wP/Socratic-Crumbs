import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Hook to handle TTS streaming via WebSockets
 * Provides functionality to stream audio from the server and play it in real-time
 */
const useTTSStreaming = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle');
  const wsRef = useRef(null);
  const clientIdRef = useRef(`client-${Date.now()}`);
  const audioQueue = useRef([]);
  const audioContextRef = useRef(null);
  
  // Initialize AudioContext on first user interaction
  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      return true;
    }
    return false;
  }, []);
  
  // Play an audio buffer
  const playAudio = useCallback((arrayBuffer) => {
    if (!audioContextRef.current) return;
    
    const audioContext = audioContextRef.current;
    
    // Create audio buffer from array buffer
    audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
      // Create audio source
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      
      // Play the audio
      source.start(0);
      
      // Process next audio in queue when this one ends
      source.onended = () => {
        if (audioQueue.current.length > 0) {
          playAudio(audioQueue.current.shift());
        }
      };
    });
  }, []);
  
  // Function to connect to TTS streaming WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    
    // Generate WebSocket URL based on current host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/tts/stream/${clientIdRef.current}`;
    
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('TTS WebSocket connected');
      setStatus('connected');
    };
    
    wsRef.current.onmessage = (event) => {
      // Handle binary data (audio)
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then(buffer => {
          if (audioQueue.current.length === 0) {
            // Play immediately if queue is empty
            playAudio(buffer);
          } else {
            // Add to queue if already playing something
            audioQueue.current.push(buffer);
          }
        });
      } 
      // Handle JSON messages (status updates)
      else {
        try {
          const data = JSON.parse(event.data);
          console.log('TTS WebSocket message:', data);
          
          if (data.status === 'error') {
            setError(data.message);
            setStatus('error');
          } else if (data.status === 'segment') {
            setStatus('streaming');
          } else if (data.status === 'started') {
            setStatus('streaming');
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      }
    };
    
    wsRef.current.onerror = (event) => {
      console.error('TTS WebSocket error:', event);
      setError('WebSocket error occurred');
      setStatus('error');
    };
    
    wsRef.current.onclose = () => {
      console.log('TTS WebSocket closed');
      setStatus('disconnected');
      setIsStreaming(false);
    };
  }, [playAudio]);
  
  // Function to start TTS streaming
  const startStreaming = useCallback((text) => {
    if (!text) {
      setError('No text provided');
      return;
    }
    
    // Initialize audio context on first interaction
    initAudioContext();
    
    // Resume AudioContext if it was suspended
    if (audioContextRef.current?.state === 'suspended') {
      audioContextRef.current.resume();
    }
    
    // Make sure WebSocket is connected
    connectWebSocket();
    
    // Reset any previous errors
    setError(null);
    setIsStreaming(true);
    
    // Wait for connection to establish before sending
    const sendMessage = () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ text }));
      } else if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
        // Wait a bit more for connection
        setTimeout(sendMessage, 100);
      } else {
        setError('WebSocket not connected');
        setIsStreaming(false);
      }
    };
    
    sendMessage();
  }, [connectWebSocket, initAudioContext]);
  
  // Function to stop TTS streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    setStatus('idle');
    
    // Clear audio queue
    audioQueue.current = [];
    
    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);
  
  return {
    isStreaming,
    error,
    status,
    startStreaming,
    stopStreaming
  };
};

export default useTTSStreaming;
