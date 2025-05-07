import React, { useState } from 'react';
import './App.css';
import 'katex/dist/katex.min.css';
import ModelSelector from './components/ModelSelector';
import MessageBubble from './components/MessageBubble';
import InputArea from './components/InputArea';
import AssistantBubble from './components/AssistantBubble';
import { useChatLogic } from './hooks/useChatLogic';
import { useModels } from './hooks/useModels';
import { useChatScroll } from './hooks/useChatScroll';

function App() {
  const [userInput, setUserInput] = useState('');

  const { models, selectedModel, setSelectedModel } = useModels();
  const {
    messages,
    isOverallStreaming,
    handleToggleThinking,
    handleStopStreaming,
    handleSubmit: handleChatSubmit,
  } = useChatLogic([], selectedModel);

  const { chatContainerRef, isUserScrolled, handleScrollToBottom } = useChatScroll([messages, isOverallStreaming]);

  const handleSubmitWrapper = async (e) => {
    e.preventDefault();
    await handleChatSubmit(userInput, setUserInput, messages, selectedModel);
  };

  return (
    <div className="App">
      <div className="header">
        <h2>Socratic Crumbs</h2>
        <ModelSelector
          models={models}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          isOverallStreaming={isOverallStreaming}
        />
      </div>

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

      <InputArea
        userInput={userInput}
        setUserInput={setUserInput}
        isOverallStreaming={isOverallStreaming}
        handleSubmit={handleSubmitWrapper}
        handleStopStreaming={handleStopStreaming}
      />
    </div>
  );
}

export default App;
