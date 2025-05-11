// Import necessary React features, styles, and components.
import React, { useState } from 'react';
import './App.css'; // Main application styles
import 'katex/dist/katex.min.css'; // Styles for KaTeX (LaTeX rendering)

// Import UI components
import ModelSelector from './components/ModelSelector';
import MessageBubble from './components/MessageBubble';
import InputArea from './components/InputArea';
import AssistantBubble from './components/AssistantBubble';

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
        <h2>Socratic Crumbs</h2>
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
