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
  const audioQueue = useRef([]);
  const isProcessing = useRef(false);

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

  const processAudioQueue = async () => {
    if (isProcessing.current || audioQueue.current.length === 0) return;
    
    const processingStart = Date.now();
    console.log(`[${processingStart}] Starting audio processing`);
    isProcessing.current = true;
    const context = new (window.AudioContext || window.webkitAudioContext)();
    setAudioContext(context);

    try {
      const audioData = audioQueue.current[0];
      console.log(`[${Date.now()}] Processing audio data, size:`, audioData.byteLength);
      
      try {
        const blobCreateStart = Date.now();
        // Convert the chunk to a Blob and create an object URL
        const blob = new Blob([audioData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        console.log(`[${Date.now()}] Blob created, time: ${Date.now() - blobCreateStart}ms`);
        
        const fetchStart = Date.now();
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        console.log(`[${Date.now()}] Audio data fetched, time: ${Date.now() - fetchStart}ms`);
        
        const decodeStart = Date.now();
        console.log(`[${decodeStart}] Starting audio buffer decode`);
        const audioBuffer = await context.decodeAudioData(arrayBuffer);
        console.log(`[${Date.now()}] Audio buffer decoded, time: ${Date.now() - decodeStart}ms`);
        
        const source = context.createBufferSource();
        source.connect(context.destination);
        source.buffer = audioBuffer;
        
        // Calculate start time based on current context time
        const startTime = context.currentTime;
        console.log(`[${Date.now()}] Starting audio playback, total processing time: ${Date.now() - processingStart}ms`);
        
        // Create a promise that resolves when the audio finishes playing
        await new Promise((resolve) => {
          source.onended = () => {
            const playbackEnd = Date.now();
            console.log(`[${playbackEnd}] Audio finished playing, total time from start: ${playbackEnd - processingStart}ms`);
            URL.revokeObjectURL(url);
            resolve();
          };
          source.start(startTime);
        });
        
      } catch (error) {
        console.error('Error processing audio:', error);
      }
      
      // Clear the queue after processing
      audioQueue.current = [];
      
    } catch (error) {
      console.error('Error in audio processing:', error);
    } finally {
      isProcessing.current = false;
      if (audioQueue.current.length === 0) {
        console.log(`[${Date.now()}] Audio processing complete, closing context`);
        context.close();
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
        const messageCompleteTime = Date.now();
        console.log(`[${messageCompleteTime}] Message complete, starting TTS processing`);

        // Process the text for TTS
        const processNewText = async () => {
          // Only process if we have text and it's different from current text
          if (newText && newText !== currentText) {
            try {
              const ttsRequestStart = Date.now();
              console.log(`[${ttsRequestStart}] Sending TTS request, delay from message complete: ${ttsRequestStart - messageCompleteTime}ms`);
              
              const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                  text: newText
                }),
              });

              if (!response.ok) {
                throw new Error(`TTS request failed: ${response.status} ${response.statusText}`);
              }

              const ttsResponseReceived = Date.now();
              console.log(`[${ttsResponseReceived}] TTS response received, processing time: ${ttsResponseReceived - ttsRequestStart}ms`);
              
              const audioData = await response.arrayBuffer();
              const audioDataReceived = Date.now();
              console.log(`[${audioDataReceived}] Audio data received, size: ${audioData.byteLength} bytes, download time: ${audioDataReceived - ttsResponseReceived}ms`);
              
              // Add to queue and process
              audioQueue.current = [audioData];
              if (!isProcessing.current) {
                console.log(`[${Date.now()}] Starting audio processing`);
                processAudioQueue();
              }
            } catch (error) {
              console.error('TTS error:', error);
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
  }, [messages, isTTSEnalbed, ttsEnabledTimestamp, currentText, isOverallStreaming]);

  // --- Event Handlers ---
  // Wrapper function for submitting the user's input
  // Prevents default form submission and calls the chat submission handler from useChatLogic
  const handleSubmitWrapper = async (e) => {
    e.preventDefault(); // Prevent page reload on form submission
    // Pass current userInput, its setter, the messages array, selected model, selected image, and its setter
    await handleChatSubmit(userInput, setUserInput, messages, selectedModel, selectedImage, setSelectedImage, setImagePreview, setMessages);
  };

  // --- JSX Rendering ---
  return (
    <div className="App"> {/* Main application container */}
      {/* Header section including the application title and model selector */}
      <div className="header">
        <div className="header-left">
          <TTSButton 
            isStreaming={isOverallStreaming}
            currentText={currentText}
            onTTSToggle={handleTTSToggle}
          />
          <h2>Socratic Crumbs</h2>
        </div>
        <ModelSelector
          models={models} // Pass the list of available models
          selectedModel={selectedModel} // Pass the currently selected model
          setSelectedModel={setSelectedModel} // Pass the function to update the selected model
          isOverallStreaming={isOverallStreaming} // Pass streaming status to disable selector during streaming
        />
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
        userInput={userInput} // Current value of the input field
        setUserInput={setUserInput} // Function to update the input field's value
        isOverallStreaming={isOverallStreaming} // Pass streaming status to disable input during streaming
        handleSubmit={handleSubmitWrapper} // Pass the submit handler wrapper
        handleStopStreaming={handleStopStreaming} // Pass the stop streaming handler
        selectedImage={selectedImage} // Pass selected image
        setSelectedImage={setSelectedImage} // Pass setter for selected image
        imagePreview={imagePreview} // Pass image preview URL
        setImagePreview={setImagePreview} // Pass setter for image preview URL
      />
    </div>
  );
}

export default App;
