import React, { useRef, useState } from 'react';

// MicButton: records audio and sends to /api/stt, then calls onResult(text)
function MicButton({ disabled, onResult }) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  const [mediaStream, setMediaStream] = useState(null);

  // Always stop and clean up the stream
  const cleanupStream = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      setMediaStream(null);
    }
  };

  // Start or stop recording on button click
  const handleMicClick = async () => {
    if (disabled) return;
    if (recording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices) return;
    // Clean up any previous stream before starting a new one
    cleanupStream();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    setMediaStream(stream);
    const mediaRecorder = new window.MediaRecorder(stream);
    audioChunksRef.current = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      // Always release the mic
      cleanupStream();
      if (audioChunksRef.current.length === 0) {
        setRecording(false);
        return;
      }
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      if (audioBlob.size === 0) {
        setRecording(false);
        return;
      }
      // Send to backend
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');
      try {
        const resp = await fetch('/api/stt', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.text && onResult) onResult(data.text);
      } catch (e) { /* handle error */ }
      setRecording(false);
    };
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
    }
    // Do NOT call cleanupStream() here; let onstop handle it
  };

  // Handle file upload for STT
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const resp = await fetch('/api/stt', { method: 'POST', body: formData });
      const data = await resp.json();
      if (data.text && onResult) onResult(data.text);
    } catch (e) { /* handle error */ }
    fileInputRef.current.value = '';
  };

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center' }}>
      <button
        type="button"
        onClick={handleMicClick}
        disabled={disabled}
        style={{
          background: 'none',
          border: 'none',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.5 : 1,
          marginRight: 4,
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
        }}
        title={recording ? 'Click to stop recording' : 'Click to start recording'}
      >
        {recording && (
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
            zIndex: 0,
          }} />
        )}
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ position: 'relative', zIndex: 1 }}>
          <path d="M12 1C11.2 1 10.44 1.32 9.88 1.88C9.32 2.44 9 3.2 9 4V12C9 12.8 9.32 13.56 9.88 14.12C10.44 14.68 11.2 15 12 15C12.8 15 13.56 14.68 14.12 14.12C14.68 13.56 15 12.8 15 12V4C15 3.2 14.68 2.44 14.12 1.88C13.56 1.32 12.8 1 12 1Z" stroke={recording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M19 10V12C19 13.86 18.26 15.64 16.95 16.95C15.64 18.26 13.86 19 12 19C10.14 19 8.36 18.26 7.05 16.95C5.74 15.64 5 13.86 5 12V10" stroke={recording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M12 19V23" stroke={recording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M8 23H16" stroke={recording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      <input
        type="file"
        ref={fileInputRef}
        accept="audio/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
        disabled={disabled}
        style={{
          background: 'none',
          border: 'none',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.5 : 1,
          color: '#888',
        }}
        title="Upload audio file for STT"
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M10 3V17M10 17L5 12M10 17L15 12" stroke="#888" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      {/* Speaker icon pulse animation keyframes (copied for mic) */}
      <style>{`
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(124,58,237,0.3); }
          70% { box-shadow: 0 0 0 10px rgba(124,58,237,0); }
          100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
        }
      `}</style>
    </div>
  );
}

export default MicButton;
