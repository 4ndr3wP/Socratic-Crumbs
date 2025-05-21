import React, { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';

// Set up PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

// InputArea component
function InputArea({
  userInput,
  setUserInput,
  isOverallStreaming,
  handleSubmit,
  handleStopStreaming,
  selectedImage,
  setSelectedImage,
  imagePreview,
  setImagePreview,
  isAudioPlaying,
  isTTSLoading
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  const [audioLevel, setAudioLevel] = useState(0);
  const audioAnalyserRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Function to analyze audio and update visual indicator
  const analyzeAudio = (stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    audioAnalyserRef.current = {
      analyser,
      dataArray,
      audioContext
    };
    
    const updateAudioLevel = () => {
      if (!audioAnalyserRef.current) return;
      
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const average = sum / bufferLength;
      // Convert to a value between 0 and 1
      const normalizedLevel = Math.min(1, average / 128);
      setAudioLevel(normalizedLevel);
      
      animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
    };
    
    updateAudioLevel();
  };

  const handleMicClick = async () => {
    console.log('Mic button clicked. isRecording:', isRecording);
    if (isRecording) {
      if (mediaRecorder) {
        console.log('Stopping mediaRecorder...');
        mediaRecorder.stop();
      }
      setIsRecording(false);
      
      // Stop audio analysis
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (audioAnalyserRef.current && audioAnalyserRef.current.audioContext) {
        audioAnalyserRef.current.audioContext.close();
        audioAnalyserRef.current = null;
      }
      setAudioLevel(0);
    } else {
      if (navigator.mediaDevices && window.MediaRecorder) {
        try {
          console.log('Requesting audio stream...');
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          
          // Start audio analysis for visualization
          analyzeAudio(stream);
          
          const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
          const recorder = new window.MediaRecorder(stream, { mimeType });
          setMediaRecorder(recorder);
          
          // Use a local variable for audio chunks
          let localAudioChunks = [];
          
          recorder.ondataavailable = (e) => {
            console.log('ondataavailable called, data size:', e.data.size);
            if (e.data.size > 0) {
              localAudioChunks.push(e.data);
            }
          };
          
          recorder.onstop = async () => {
            console.log('Recorder stopped. localAudioChunks:', localAudioChunks.length);
            if (localAudioChunks.length === 0) {
              console.error('No audio data captured');
              setIsUploading(false);
              return;
            }
            
            const audioBlob = new Blob(localAudioChunks, { type: mimeType });
            console.log('Audio Blob type:', audioBlob.type, 'size:', audioBlob.size);
            setIsUploading(true);
            try {
              const formData = new FormData();
              formData.append('file', audioBlob, 'recording.webm'); // use .webm extension
              console.log('Uploading audio to STT server...');
              const response = await fetch('/api/stt', {
                method: 'POST',
                body: formData,
              });
              console.log('STT server response status:', response.status);
              const data = await response.json(); // Parse JSON
              console.log('STT server response data:', data);
              if (data && data.text) {
                setUserInput(prev => prev ? prev + ' ' + data.text : data.text);
              } else if (data && data.error) {
                setUserInput(prev => prev ? prev + ' [STT error: ' + data.error + ']' : '[STT error: ' + data.error + ']');
              }
            } catch (err) {
              console.error('STT upload failed', err);
            } finally {
              setIsUploading(false);
            }
          };
          recorder.start();
          setIsRecording(true);
          console.log('Recording started.');
        } catch (err) {
          console.error('Could not start audio recording', err);
        }
      } else {
        alert('Audio recording not supported in this browser.');
      }
    }
  };

  const handleInputChange = (e) => {
    const textarea = e.target;
    setUserInput(textarea.value);

    // Reset height to auto to ensure it shrinks correctly
    textarea.style.height = 'auto';

    const computedStyle = getComputedStyle(textarea);
    const lineHeight = parseFloat(computedStyle.lineHeight);
    const paddingTop = parseFloat(computedStyle.paddingTop);
    const paddingBottom = parseFloat(computedStyle.paddingBottom);
    const borderTop = parseFloat(computedStyle.borderTopWidth);
    const borderBottom = parseFloat(computedStyle.borderBottomWidth);

    const MAX_LINES = 5;
    const maxHeight = (lineHeight * MAX_LINES) + paddingTop + paddingBottom + borderTop + borderBottom;
    const singleLineHeight = lineHeight + paddingTop + paddingBottom + borderTop + borderBottom;

    // Calculate the scroll height (content height)
    const scrollHeight = textarea.scrollHeight;

    if (scrollHeight <= singleLineHeight) {
        // If content is less than or fits one line, set to auto (or explicit single line height)
        textarea.style.height = 'auto'; // Let CSS handle the min-height for single line
        textarea.style.overflowY = 'hidden';
    } else if (scrollHeight <= maxHeight) {
        // If content is more than one line but less than max, adjust height
        textarea.style.height = `${scrollHeight}px`;
        textarea.style.overflowY = 'hidden';
    } else {
        // If content exceeds max height, set to max height and show scrollbar
        textarea.style.height = `${maxHeight}px`;
        textarea.style.overflowY = 'auto';
    }
  };

  useEffect(() => {
    // When userInput becomes empty (e.g., after sending a message),
    // reset the textarea height to its initial single-line state.
    if (userInput === '' && textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // Rely on CSS min-height
      textareaRef.current.style.overflowY = 'hidden';
    }
  }, [userInput]);

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      if (file.type === 'application/pdf') {
        // Extract text from PDF
        let extractedText = '';
        try {
          const arrayBuffer = await file.arrayBuffer();
          const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
          const numPages = pdf.numPages;
          for (let i = 1; i <= numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items.map(item => item.str).join(' ');
            extractedText += pageText + '\n';
          }
        } catch (err) {
          console.error('Failed to extract PDF text:', err);
        }
        setImagePreview({
          isPdf: true,
          originalName: file.name,
          pdfText: extractedText
        });
      } else {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview({
            url: reader.result,
            isPdf: false
          });
        };
        reader.readAsDataURL(file);
      }
    } else {
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const onFormSubmit = (e) => {
    e.preventDefault();
    if (isOverallStreaming || isAudioPlaying) {
      handleStopStreaming();
    } else {
      handleSubmit(e);
    }
  };

  const isBusy = isOverallStreaming || isAudioPlaying || isTTSLoading;
  const hasText = userInput.trim().length > 0;

  // Button color logic
  let buttonBg = '#f3f4f6'; // default grey
  let buttonColor = '#9ca3af'; // default grey text
  let buttonBorder = 'none';
  if (isBusy) {
    buttonBg = '#fff'; // white for Stop
    buttonColor = '#333'; // dark text
    buttonBorder = '1px solid #ddd'; // subtle border
  } else if (hasText) {
    buttonBg = '#2563eb'; // blue-600 (matches prompt bubble)
    buttonColor = 'white';
    buttonBorder = 'none';
  }

  return (
    <form onSubmit={onFormSubmit} className="input-area" style={{ width: '100%', padding: '16px', background: '#fff', borderTop: '1px solid #eee' }}>
      <div className="input-container" style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
        {/* Attachment icon */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', height: '36px' }}>
          <button
            type="button"
            onClick={selectedImage ? () => { setSelectedImage(null); setImagePreview(null); if(fileInputRef.current) fileInputRef.current.value = null; } : triggerFileInput}
            className="attach-image-button"
            disabled={isBusy}
            style={{
              background: 'none',
              border: 'none',
              padding: 0,
              marginRight: '4px',
              cursor: isBusy ? 'not-allowed' : 'pointer',
              opacity: isBusy ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              height: '36px',
              color: selectedImage ? '#2563eb' : '#333',
              position: 'relative',
              zIndex: 1
            }}
            title={selectedImage ? 'Remove attachment' : 'Attach image'}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M16 13H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M16 17H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M10 9H9H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            {selectedImage && (
              <span
                style={{
                  position: 'absolute',
                  top: '-4px',
                  right: '-4px',
                  background: '#f3f4f6',
                  color: '#888',
                  borderRadius: '50%',
                  width: '16px',
                  height: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '13px',
                  fontWeight: 700,
                  boxShadow: '0 1px 2px rgba(0,0,0,0.07)',
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  zIndex: 2
                }}
                onClick={e => {
                  e.stopPropagation();
                  setSelectedImage(null);
                  setImagePreview(null);
                  if(fileInputRef.current) fileInputRef.current.value = null;
                }}
                title="Remove attachment"
              >
                ×
              </span>
            )}
          </button>
        </div>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*,.pdf"
          onChange={handleImageChange}
          disabled={isBusy}
        />
        {/* Text box with mic button */}
        <div style={{ position: 'relative', flex: 1 }}>
          <textarea
            ref={textareaRef}
            value={userInput}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if ((userInput.trim() || selectedImage) && e.target.form) {
                  e.target.form.requestSubmit();
                }
              }
            }}
            placeholder="Type your message..."
            disabled={isBusy}
            rows={1}
            style={{
              resize: 'none',
              width: '100%',
              minHeight: '36px',
              maxHeight: '120px',
              padding: '8px 12px',
              paddingRight: '40px', // Make room for mic button
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              backgroundColor: '#fff',
              fontSize: '15px',
              lineHeight: '1.5',
              color: '#1f2937',
              outline: 'none',
              transition: 'border-color 0.2s',
              opacity: isBusy ? 0.7 : 1,
              cursor: isBusy ? 'not-allowed' : 'text',
              marginRight: '8px',
            }}
          />            <button
              type="button"
              onClick={handleMicClick}
              disabled={isBusy || isUploading}
              style={{
                position: 'absolute',
                right: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'none',
                border: 'none',
                padding: '4px',
                cursor: isBusy || isUploading ? 'not-allowed' : 'pointer',
                opacity: isBusy || isUploading ? 0.5 : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 2,
              }}
              title={isRecording ? 'Stop recording' : (isUploading ? 'Transcribing...' : 'Start recording')}
            >
              {isRecording && (
                <>
                  {/* Background pulse animation */}
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    animation: 'pulse 2s infinite',
                    zIndex: 0,
                  }} />
                  
                  {/* Audio level visualizer rings */}
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: `${24 + audioLevel * 16}px`,
                    height: `${24 + audioLevel * 16}px`, 
                    borderRadius: '50%',
                    border: `2px solid rgba(124, 58, 237, ${0.3 + audioLevel * 0.7})`,
                    transition: 'all 0.1s ease',
                    zIndex: 0,
                  }} />
                  
                  {/* Text indicator - optional */}
                  <div style={{
                    position: 'absolute',
                    top: '-20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: '12px',
                    color: '#7C3AED',
                    fontWeight: 'bold',
                    whiteSpace: 'nowrap',
                    textShadow: '0 0 4px white, 0 0 4px white, 0 0 4px white, 0 0 4px white',
                    animation: 'fadeInOut 2s infinite',
                    zIndex: 3,
                  }}>
                    Recording...
                  </div>
                </>
              )}
              {isUploading ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ position: 'relative', zIndex: 1 }}>
                  <circle cx="12" cy="12" r="10" stroke="#7C3AED" strokeWidth="3" fill="none" strokeDasharray="60" strokeDashoffset="30">
                    <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
                  </circle>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ position: 'relative', zIndex: 1 }}>
                  <path d="M12 1C11.2 1 10.44 1.32 9.88 1.88C9.32 2.44 9 3.2 9 4V12C9 12.8 9.32 13.56 9.88 14.12C10.44 14.68 11.2 15 12 15C12.8 15 13.56 14.68 14.12 14.12C14.68 13.56 15 12.8 15 12V4C15 3.2 14.68 2.44 14.12 1.88C13.56 1.32 12.8 1 12 1Z" stroke={isRecording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M19 10V12C19 13.86 18.26 15.64 16.95 16.95C15.64 18.26 13.86 19 12 19C10.14 19 8.36 18.26 7.05 16.95C5.74 15.64 5 13.86 5 12V10" stroke={isRecording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 19V23" stroke={isRecording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M8 23H16" stroke={isRecording ? '#7C3AED' : '#888'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
          </button>
        </div>
        {/* Send/Stop button */}
        <button
          type="submit"
          disabled={isBusy ? false : !hasText}
          style={{
            padding: '8px 22px',
            backgroundColor: buttonBg,
            color: buttonColor,
            border: buttonBorder,
            borderRadius: '7px',
            cursor: isBusy ? 'pointer' : (!hasText ? 'not-allowed' : 'pointer'),
            fontSize: '15px',
            fontWeight: 600,
            transition: 'all 0.2s',
            boxShadow: '0 1px 2px rgba(0,0,0,0.03)'
          }}
        >
          {isBusy ? 'Stop' : 'Send'}
        </button>
      </div>
      <style>
        {`
          @keyframes pulse {
            0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.8; }
            50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.4; }
            100% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.8; }
          }
          
          @keyframes fadeInOut {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
          }
          
          .recording-wave {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            z-index: 0;
          }
        `}
      </style>
    </form>
  );
}

export default InputArea;