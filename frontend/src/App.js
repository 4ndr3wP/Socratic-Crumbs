import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './App.css'; 
import ModelSelector from './components/ModelSelector'; 
import MessageBubble from './components/MessageBubble'; 
import InputArea from './components/InputArea'; 
import AssistantBubble from './components/AssistantBubble'; // Import AssistantBubble

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isOverallStreaming, setIsOverallStreaming] = useState(false); 
  const [isUserScrolled, setIsUserScrolled] = useState(false);
  const chatContainerRef = useRef(null);
  const [abortController, setAbortController] = useState(null); 

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setModels(data.models);
          if (data.models.length > 0) {
            setSelectedModel(data.models[0]);
          }
        }
      })
      .catch(err => console.error('Failed to load models:', err));
  }, []);
  
  useEffect(() => {
    const chatDiv = chatContainerRef.current;
    if (chatDiv) {
      const handleScrollEvent = () => { 
        const isAtBottom =
          Math.abs(chatDiv.scrollHeight - chatDiv.scrollTop - chatDiv.clientHeight) < 1.5; 
        setIsUserScrolled(!isAtBottom); 
      };
  
      chatDiv.addEventListener('scroll', handleScrollEvent);
      return () => chatDiv.removeEventListener('scroll', handleScrollEvent);
    }
  }, []);
  
  useEffect(() => {
    const chatDiv = chatContainerRef.current;
    if (chatDiv && !isUserScrolled) {
      chatDiv.scrollTop = chatDiv.scrollHeight; 
    }
  }, [messages, isOverallStreaming, isUserScrolled]);
  
  const handleScrollToBottom = () => {
    const chatDiv = chatContainerRef.current;
    if (chatDiv) {
      chatDiv.scrollTop = chatDiv.scrollHeight;
      setIsUserScrolled(false); 
    }
  };

  const handleStopStreaming = () => {
    if (abortController) {
      abortController.abort();
      console.log("Streaming stop requested by user.");
    }
  };

  const handleToggleThinking = (messageId) => {
    setMessages(prevMessages =>
      prevMessages.map(msg =>
        msg.id === messageId ? { ...msg, isThinkingVisible: !msg.isThinkingVisible } : msg
      )
    );
  };

  const parseThinkResponse = (rawContent) => {
    const thinkTagStart = "<think>";
    const thinkTagEnd = "</think>";
    let thinkingText = null;
    let responseText = rawContent || ""; 
    let showToggleButton = false;

    const thinkStartIndex = responseText.indexOf(thinkTagStart);
    const thinkEndIndex = responseText.indexOf(thinkTagEnd);

    if (thinkStartIndex !== -1 && thinkEndIndex > thinkStartIndex) {
      thinkingText = responseText.substring(thinkStartIndex + thinkTagStart.length, thinkEndIndex).trim();
      responseText = (responseText.substring(0, thinkStartIndex) + responseText.substring(thinkEndIndex + thinkTagEnd.length)).trim();
      showToggleButton = true;
    }
    return { thinkingText, responseText, showToggleButton };
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isOverallStreaming) return;

    const newUserMessage = { role: 'user', content: userInput, id: Date.now() + '-user' };
    setMessages(prev => [...prev, newUserMessage]);
    setUserInput('');

    const conversationPayload = messages
      .filter(msg => msg.role === 'user' || (msg.role === 'assistant' && msg.isStreamingComplete))
      .map(m => ({ role: m.role, content: m.role === 'assistant' ? m.response : m.content })) 
      .concat([{ role: 'user', content: userInput }]); 
    
    const controller = new AbortController(); 
    setAbortController(controller); 

    const assistantMsgId = Date.now() + '-assistant'; // Use this local const to identify the message

    setMessages(prev => [...prev, { 
      role: 'assistant', 
      id: assistantMsgId, 
      content: '', 
      thinking: null,
      response: null,
      isStreamingThisMessage: true, 
      isStreamingComplete: false, 
      isThinkingVisible: false,
      showToggle: false
    }]);
    setIsOverallStreaming(true); 

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, messages: conversationPayload }), 
        signal: controller.signal, 
      });
      if (!response.body) throw new Error('No response body');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        if (value) {
          const decodedChunk = decoder.decode(value, { stream: true }); 
          setMessages(prevMessages =>
            prevMessages.map(msg =>
              msg.id === assistantMsgId // Use local const here
                ? { ...msg, content: msg.content + decodedChunk } 
                : msg
            )
          );
        }
        done = doneReading;
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Streaming aborted.');
         setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) { // Use local const here
                const finalContent = (msg.content || "") + "\n*Streaming stopped.*";
                const { thinkingText, responseText, showToggleButton } = parseThinkResponse(finalContent);
                return {
                    ...msg, 
                    content: finalContent, 
                    thinking: thinkingText,
                    response: responseText,
                    isStreamingThisMessage: false,
                    isStreamingComplete: true, 
                    showToggle: showToggleButton,
                };
            }
            return msg;
         }));
      } else {
        console.error('Error during fetch:', err);
        setMessages(prev => prev.map(msg => {
            if (msg.id === assistantMsgId) { // Use local const here
                const errorContent = '*Error: Failed to get response*';
                return {
                    ...msg, 
                    content: errorContent,
                    thinking: null,
                    response: errorContent,
                    isStreamingThisMessage: false,
                    isStreamingComplete: true,
                };
            }
            return msg;
        }));
      }
    } finally {
      setIsOverallStreaming(false); 
      setAbortController(null);
      
      setMessages(prevMessages =>
        prevMessages.map(msg => {
          if (msg.id === assistantMsgId && !msg.isStreamingComplete) { // Use local const here
            const { thinkingText, responseText, showToggleButton } = parseThinkResponse(msg.content);
            return { 
              ...msg, 
              thinking: thinkingText, 
              response: responseText, 
              isStreamingThisMessage: false, 
              isStreamingComplete: true,
              showToggle: showToggleButton,
            };
          }
          return msg;
        })
      );
    }
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
        handleSubmit={handleSubmit}
        handleStopStreaming={handleStopStreaming}
      />
    </div>
  );
}

export default App;
