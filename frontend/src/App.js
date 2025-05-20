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

  // --- Event Handlers ---
  // Wrapper function for submitting the user's input
  // Prevents default form submission and calls the chat submission handler from useChatLogic
  const handleSubmitWrapper = async (e) => {
    e.preventDefault();
    await handleChatSubmit(userInput, setUserInput, messages, selectedModel, selectedImage, setSelectedImage, setImagePreview, setMessages);
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

  // --- JSX Rendering ---
  return (
    <div className="App"> {/* Main application container */}
      {/* Header section including the application title and model selector */}
      <div className="header">
        <div className="header-content" style={{ flexDirection: 'column', display: 'flex', alignItems: 'center', width: '100%' }}>
          <div className="header-top" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px', width: '100%' }}>
            <TTSButton 
              isStreaming={isOverallStreaming}
              currentText={currentText}
              onTTSToggle={handleTTSToggle}
              style={{ marginRight: '20px' }}
            />
            <h2 style={{ margin: '0 20px', padding: 0, minWidth: 'max-content', fontSize: '2rem', fontWeight: 700, letterSpacing: '0.5px' }}>Socratic Crumbs</h2>
            <MicButton
              disabled={isOverallStreaming || isAudioPlaying || isTTSLoading}
              onResult={text => setUserInput(prev => prev ? prev + ' ' + text : text)}
            />
          </div>
          <div className="header-bottom" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', marginTop: '8px' }}>
            <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-start' }}>
              <VoiceSelector
                voices={voices}
                selectedVoice={selectedVoice}
                setSelectedVoice={setSelectedVoice}
                isStreaming={isOverallStreaming}
              />
            </div>
            <div style={{ flex: 1 }}></div>
            <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
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

      {/* Chat messages container, ref is managed by useChatScroll */}
      <div className="chat-container" ref={chatContainerRef}>
        {/* Map through messages and render a MessageBubble for each */}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id} // Unique key for each message
            msg={msg} // The message object itself
            onToggleThinking={handleToggleThinking} // Pass the toggle thinking handler
            AssistantBubbleComponent={AssistantBubble} // Pass the component to render assistant messages
          />
        ))}
        {/* Display a scroll-to-bottom button if the user has scrolled up */}
        {isUserScrolled && (
          <button
            className="scroll-to-bottom"
            onClick={handleScrollToBottom} // Scroll to bottom when clicked
          >
            ↓
          </button>
        )}
      </div>

      {/* Input area for the user to type messages */}
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
    </div>
  );
}

export default App;
