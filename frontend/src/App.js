// Import necessary React features, styles, and components.
import React, { useState, useEffect, useRef } from 'react';
import './App.css'; // Main application styles
import 'katex/dist/katex.min.css'; // Styles for KaTeX (LaTeX rendering)

// Import UI components
import ModelSelector from './components/ModelSelector';
import MessageBubble from './components/MessageBubble';
import InputArea from './components/InputArea';
import AssistantBubble from './components/AssistantBubble';
import TTSButton from './components/TTSButton';
import VoiceSelector from './components/VoiceSelector';
import MicButton from './components/MicButton';
import STSButton from './components/STSButton'; // Import the new STS Button component
import STSVisualizer from './components/STSVisualizer';

// Import custom hooks to manage specific functionalities
import { useChatLogic } from './hooks/useChatLogic'; // Manages chat message state and API interactions
import { useModels } from './hooks/useModels';       // Manages fetching and selection of AI models
import { useChatScroll } from './hooks/useChatScroll'; // Manages scroll behavior of the chat container

// Main application component
function App() {
  // State for the user's current input in the text area
  const [userInput, setUserInput] = useState('');
  const [selectedImage, setSelectedImage] = useState(null); // New state for selected image file
  const [imagePreview, setImagePreview] = useState(null); // New state for image preview URL
  const [currentText, setCurrentText] = useState(''); // New state for TTS text
  const [isTTSEnalbed, setIsTTSEnalbed] = useState(false);
  const [ttsEnabledTimestamp, setTtsEnabledTimestamp] = useState(null);
  const [audioContext, setAudioContext] = useState(null);
  const [selectedVoice, setSelectedVoice] = useState('af_heart');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const audioQueue = useRef([]);
  const isProcessing = useRef(false);
  const [abortController, setAbortController] = useState(null);
  const [isTTSLoading, setIsTTSLoading] = useState(false);
  // STS mode state
  const [stsActive, setStsActive] = useState(false);
  const [stsStatus, setStsStatus] = useState('idle');
  const [stsMessages, setStsMessages] = useState([]);
  const [assistantAudioData, setAssistantAudioData] = useState([]);
  const [userAudioData, setUserAudioData] = useState([]);
  const [userTranscript, setUserTranscript] = useState('');
  const [assistantTranscript, setAssistantTranscript] = useState('');
  const [assistantAudioSourceRef, setAssistantAudioSourceRef] = useState(null);
  const [assistantAudioContext, setAssistantAudioContext] = useState(null);
  const [assistantAudioBuffer, setAssistantAudioBuffer] = useState(null);
  const audioFileInputRef = useRef(null);

  // --- Custom Hook Integrations ---
  // Manage AI model data (list of models, selected model, and setter)
  const { models, selectedModel, setSelectedModel } = useModels();

  // Manage chat logic (messages, streaming status, and handlers for chat actions)
  // Pass the currently selected model to the chat logic hook
  const {
    messages,               // Array of chat messages
    setMessages,            // Allow App.js to update messages for image display
    isOverallStreaming,     // Boolean indicating if a response is currently streaming
    handleToggleThinking,   // Function to toggle visibility of assistant's thought process
    handleStopStreaming,    // Function to stop an ongoing stream
    handleSubmit: handleChatSubmit, // Renamed to avoid conflict, handles sending a new message
  } = useChatLogic([], selectedModel);

  // Manage chat container's scroll behavior (ref, user scroll status, and scroll-to-bottom handler)
  // Dependencies ([messages, isOverallStreaming]) trigger auto-scroll when new messages or streaming occurs
  const { chatContainerRef, isUserScrolled, handleScrollToBottom } = useChatScroll([messages, isOverallStreaming]);

  // Available TTS voices
  const voices = [
    'af_bella',
    'af_heart',
    'af_nicole',
    'im_nicola'
  ];

  const processAudioQueue = async (abortSignal) => {
    if (isProcessing.current || audioQueue.current.length === 0) return;
    setIsTTSLoading(false); // Audio is about to play, TTS loading is done
    isProcessing.current = true;
    const context = new (window.AudioContext || window.webkitAudioContext)();
    setAudioContext(context);
    setIsAudioPlaying(true);
    try {
      const audioData = audioQueue.current[0];
      if (abortSignal && abortSignal.aborted) {
        setIsAudioPlaying(false);
        return;
      }
      const blob = new Blob([audioData], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      if (abortSignal && abortSignal.aborted) {
        setIsAudioPlaying(false);
        return;
      }
      const audioBuffer = await context.decodeAudioData(arrayBuffer);
      const source = context.createBufferSource();
      source.connect(context.destination);
      source.buffer = audioBuffer;
      const startTime = context.currentTime;
      await new Promise((resolve) => {
        source.onended = () => {
          URL.revokeObjectURL(url);
          setIsAudioPlaying(false);
          resolve();
        };
        if (abortSignal && abortSignal.aborted) {
          setIsAudioPlaying(false);
          resolve();
        } else {
          source.start(startTime);
        }
      });
    } catch (error) {
      setIsAudioPlaying(false);
    } finally {
      isProcessing.current = false;
      if (audioQueue.current.length === 0 && audioContext) {
        audioContext.close();
        setAudioContext(null);
      }
    }
  };

  const handleTTSToggle = (enabled) => {
    setIsTTSEnalbed(enabled);
    if (enabled) {
      // Set timestamp when TTS is enabled
      setTtsEnabledTimestamp(Date.now());
    } else {
      // Clean up audio if TTS is disabled
      if (audioContext) {
        audioContext.close();
        setAudioContext(null);
      }
      audioQueue.current = [];
      isProcessing.current = false;
      setTtsEnabledTimestamp(null);
    }
  };

  // Update currentText and handle TTS when messages change
  useEffect(() => {
    console.log('Messages updated:', messages);
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      console.log('Last message:', lastMessage);
      
      // Process assistant messages that are complete
      // and were created after TTS was enabled
      if (lastMessage.role === 'assistant' && 
          isTTSEnalbed && 
          ttsEnabledTimestamp && 
          lastMessage.id && 
          parseInt(lastMessage.id.split('-')[0]) > ttsEnabledTimestamp &&
          !isOverallStreaming) {  // Only process when streaming is complete
        
        const newText = lastMessage.content;
        console.log(`[${Date.now()}] Message complete, starting TTS processing`);

        // Process the text for TTS
        const processNewText = async () => {
          if (newText && newText !== currentText) {
            let controller = new AbortController();
            setAbortController(controller);
            setIsTTSLoading(true); // TTS fetch is starting
            try {
              const response = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newText, voice: selectedVoice, is_streaming: false }),
                signal: controller.signal
              });
              if (!response.ok) throw new Error(`TTS request failed: ${response.status} ${response.statusText}`);
              const audioData = await response.arrayBuffer();
              if (controller.signal.aborted) return;
              audioQueue.current = [audioData];
              if (!isProcessing.current) processAudioQueue(controller.signal);
            } catch (error) {
              if (controller.signal.aborted) {
                // Aborted by user
              } else {
                console.error('TTS error:', error);
              }
            } finally {
              setAbortController(null);
              setIsTTSLoading(false); // TTS fetch is done or aborted
            }
          }
        };
        processNewText();
        setCurrentText(newText);
      }
    } else {
      console.log('No messages, clearing currentText');
      setCurrentText('');
    }
  }, [messages, isTTSEnalbed, ttsEnabledTimestamp, currentText, isOverallStreaming, selectedVoice]);

  // Extract transcription from last STS message
  useEffect(() => {
    if (stsMessages && stsMessages.length > 0) {
      const last = stsMessages[stsMessages.length - 1];
      if (last.type !== 'error' && last.text && last.text.toLowerCase().startsWith('transcribed:')) {
        setUserTranscript(last.text.replace(/^transcribed:/i, '').trim());
      }
    }
  }, [stsMessages]);

  // Real-time user microphone waveform for STS mode
  useEffect(() => {
    let audioContext, analyser, dataArray, source, rafId, stream;
    if (stsActive) {
      navigator.mediaDevices.getUserMedia({ audio: true }).then(s => {
        stream = s;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 128;
        dataArray = new Float32Array(analyser.fftSize);
        const update = () => {
          analyser.getFloatTimeDomainData(dataArray);
          setUserAudioData(Array.from(dataArray));
          rafId = requestAnimationFrame(update);
        };
        update();
      });
    }
    return () => {
      if (audioContext) audioContext.close();
      if (rafId) cancelAnimationFrame(rafId);
      if (stream) stream.getTracks().forEach(track => track.stop());
      setUserAudioData([]);
    };
  }, [stsActive]);

  // Real-time assistant waveform visualization for TTS playback
  useEffect(() => {
    let audioContext, analyser, rafId, source;
    if (stsActive && assistantAudioBuffer) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      setAssistantAudioContext(audioContext);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      source = audioContext.createBufferSource();
      source.buffer = assistantAudioBuffer;
      source.connect(analyser);
      analyser.connect(audioContext.destination);
      const dataArray = new Float32Array(analyser.fftSize);
      const update = () => {
        analyser.getFloatTimeDomainData(dataArray);
        setAssistantAudioData(Array.from(dataArray));
        rafId = requestAnimationFrame(update);
      };
      source.onended = () => {
        setAssistantAudioData([]);
        if (audioContext) audioContext.close();
      };
      source.start();
      update();
    }
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      setAssistantAudioData([]);
      if (audioContext) audioContext.close();
    };
  }, [stsActive, assistantAudioBuffer]);

  // Track last user and assistant utterances for split text
  useEffect(() => {
    if (stsMessages && stsMessages.length > 0) {
      const last = stsMessages[stsMessages.length - 1];
      if (last.type !== 'error' && last.text) {
        if (last.text.toLowerCase().startsWith('transcribed:')) {
          setUserTranscript(last.text.replace(/^transcribed:/i, '').trim());
        }
        if (last.text.toLowerCase().startsWith('response:')) {
          setAssistantTranscript(last.text.replace(/^response:/i, '').trim());
        }
      }
    }
  }, [stsMessages]);

  // Set app and body background to black when STS is active
  useEffect(() => {
    if (stsActive) {
      document.body.style.background = 'black';
    } else {
      document.body.style.background = '';
    }
    return () => { document.body.style.background = ''; };
  }, [stsActive]);

  // --- Event Handlers ---
  // Wrapper function for submitting the user's input
  // Prevents default form submission and calls the chat submission handler from useChatLogic
  const handleSubmitWrapper = async (e) => {
    e.preventDefault();
    await handleChatSubmit(userInput, setUserInput, messages, selectedModel, selectedImage, setSelectedImage, setImagePreview, setMessages, imagePreview); // Pass imagePreview
  };

  const handleStopAll = () => {
    handleStopStreaming();
    if (abortController) abortController.abort();
    if (audioContext) {
      audioContext.close();
      setAudioContext(null);
    }
    audioQueue.current = [];
    isProcessing.current = false;
    setIsAudioPlaying(false);
    setIsTTSLoading(false);
  };

  const handleAudioFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // TODO: Implement audio file upload/processing logic here
      // For now, just log the file
      console.log('Audio file selected:', file);
    }
  };

  // --- JSX Rendering ---
  return (
    <div className="App" style={{ background: stsActive ? 'black' : '#fff' }}> {/* Main application container */}
      {/* Header section including the application title and model selector */}
      <div className="header">
        <div className="header-content">
          <div className="header-top">
            <TTSButton 
              isStreaming={isOverallStreaming}
              currentText={currentText}
              onTTSToggle={handleTTSToggle}
              style={{ marginRight: '32px' }}
            />
            <h2 className="app-title">
              Socratic Crumbs
              <button
                type="button"
                onClick={() => audioFileInputRef.current && audioFileInputRef.current.click()}
                className="upload-audio-btn"
                style={{ marginLeft: '32px' }}
                title="Upload audio file"
                onMouseDown={e => e.currentTarget.style.color = '#7c3aed'}
                onMouseUp={e => e.currentTarget.style.color = '#888'}
                onMouseLeave={e => e.currentTarget.style.color = '#888'}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 16V4M12 16L8 12M12 16L16 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <rect x="4" y="18" width="16" height="2" rx="1" fill="currentColor"/>
                </svg>
              </button>
              <input
                ref={audioFileInputRef}
                type="file"
                accept="audio/*"
                style={{ display: 'none' }}
                onChange={handleAudioFileUpload}
              />
            </h2>
          </div>
          <div className="header-bottom">
            <div className="header-col left">
              <VoiceSelector
                voices={voices}
                selectedVoice={selectedVoice}
                setSelectedVoice={setSelectedVoice}
                isStreaming={isOverallStreaming}
              />
            </div>
            <div className="header-col center">
              <STSButton 
                selectedModel={selectedModel} 
                selectedVoice={selectedVoice} 
                onSTSActiveChange={setStsActive}
                onStatusChange={setStsStatus}
                onMessagesChange={setStsMessages}
                isActive={stsActive}
              />
            </div>
            <div className="header-col right">
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                isOverallStreaming={isOverallStreaming}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main area: show STSVisualizer if STS is active, else normal chat UI */}
      <div style={{ height: 400, margin: '40px 0', background: stsActive ? 'black' : 'transparent' }}>
        {stsActive ? (
          <>
            <STSVisualizer
              assistantAudioData={assistantAudioData}
              userAudioData={userAudioData}
              assistantTranscript={assistantTranscript}
              userTranscript={userTranscript}
            />
            <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, width: '100%' }}>
              <InputArea
                userInput={userInput}
                setUserInput={() => {}}
                isOverallStreaming={true}
                handleSubmit={() => {}}
                handleStopStreaming={() => setStsActive(false)}
                selectedImage={null}
                setSelectedImage={() => {}}
                imagePreview={null}
                setImagePreview={() => {}}
                isAudioPlaying={true}
                isTTSLoading={true}
                disabled={true}
                stopButtonColor="#ef4444"
              />
            </div>
          </>
        ) : (
          <div className="chat-container" ref={chatContainerRef}>
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                onToggleThinking={handleToggleThinking}
                AssistantBubbleComponent={AssistantBubble}
              />
            ))}
            {isUserScrolled && (
              <button
                className="scroll-to-bottom"
                onClick={handleScrollToBottom}
              >
                ↓
              </button>
            )}
          </div>
        )}
      </div>

      {/* Input area for the user to type messages */}
      {!stsActive && (
        <InputArea
          userInput={userInput}
          setUserInput={setUserInput}
          isOverallStreaming={isOverallStreaming || isAudioPlaying || isTTSLoading}
          handleSubmit={handleSubmitWrapper}
          handleStopStreaming={handleStopAll}
          selectedImage={selectedImage}
          setSelectedImage={setSelectedImage}
          imagePreview={imagePreview}
          setImagePreview={setImagePreview}
          isAudioPlaying={isAudioPlaying}
          isTTSLoading={isTTSLoading}
        />
      )}
    </div>
  );
}

export default App;
