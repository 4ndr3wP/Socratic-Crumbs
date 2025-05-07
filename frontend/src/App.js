import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './App.css'; 
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism'; 


function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isOverallStreaming, setIsOverallStreaming] = useState(false); 
  const [isUserScrolled, setIsUserScrolled] = useState(false);
  const chatContainerRef = useRef(null);
  const [abortController, setAbortController] = useState(null); 
  const textareaRef = useRef(null); // Ref for the textarea

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
        <div className="model-selector">
          <label htmlFor="model">Model:</label>
          <select 
            id="model" 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isOverallStreaming} 
          >
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="chat-container" ref={chatContainerRef}>
      {messages.map((msg) => ( 
        <div key={msg.id} className={`message ${msg.role}`}>
          <div className="bubble">
            {msg.role === 'assistant' ? (
              (msg.isStreamingThisMessage && !msg.content) ? 
                <div className="typing-indicator">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              : 
                <AssistantBubble 
                  message={msg} 
                  onToggleThinking={handleToggleThinking} 
                />
            ) : (
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
        </div>
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

      <form className="input-area" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef} // Add ref to textarea
          value={userInput}
          onChange={handleInputChange}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault(); // Prevent newline in textarea
              if (userInput.trim() && e.target.form) {
                e.target.form.requestSubmit(); // Trigger form submission
              }
            }
          }}
          placeholder="Type your message..."
          disabled={isOverallStreaming}
          rows="1" // Start with one row height
          style={{ overflowY: 'hidden' }} // Initially hide scrollbar, JS manages it
        />
        {!isOverallStreaming ? (
          <button type="submit" disabled={!userInput.trim()}>Send</button>
        ) : (
          <button type="button" onClick={handleStopStreaming}>Stop</button>
        )}
      </form>
    </div>
  );
}

function AssistantBubble({ message, onToggleThinking }) {
  const { 
    id, 
    content, 
    thinking, 
    response, 
    isStreamingComplete, 
    isThinkingVisible, 
    showToggle 
  } = message;

  const handleCopy = (code) => {
    navigator.clipboard.writeText(code);
  };

  if (!isStreamingComplete && content) { 
    return <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{content}</ReactMarkdown>;
  }
  
  if (isStreamingComplete) {
    return (
      <div>
        {showToggle && (
          <button onClick={() => onToggleThinking(id)} className="think-toggle-button" aria-label="Toggle Thoughts">
            🧠 
          </button>
        )}
        {isThinkingVisible && thinking && (
          <div className="thinking-bubble"> 
            <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
              {thinking}
            </ReactMarkdown>
          </div>
        )}
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              const lang = match ? match[1].toLowerCase() : null;

              if (!inline && lang && lang !== 'latex') {
                return (
                  <div style={{ position: 'relative', marginTop: '1em', marginBottom: '1em' }}>
                    <SyntaxHighlighter
                      style={dracula}
                      language={lang}
                      PreTag="div" 
                      customStyle={{
                        backgroundColor: '#000', borderRadius: '8px', paddingTop: '16px',
                        paddingBottom: '16px', paddingLeft: '16px', paddingRight: '60px',
                        fontSize: '14px', lineHeight: '1.5', overflowX: 'auto',
                      }}
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                    <button
                      onClick={() => handleCopy(String(children).replace(/\n$/, ''))}
                      style={{
                        position: 'absolute', top: '8px', right: '8px', background: '#007AFF',
                        border: 'none', borderRadius: '4px', color: '#fff', padding: '4px 8px',
                        fontSize: '12px', cursor: 'pointer', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
                      }}
                    >
                      Copy
                    </button>
                  </div>
                );
              }
              return <code className={className} {...props}>{children}</code>;
            }
          }}
        >
          {response || ""} 
        </ReactMarkdown>
      </div>
    );
  }
  return null; 
}

export default App;
