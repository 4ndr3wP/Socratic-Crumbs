import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// MessageBubble component
function MessageBubble({ msg, onToggleThinking, AssistantBubbleComponent }) {
  return (
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
            <AssistantBubbleComponent 
              message={msg} 
              onToggleThinking={onToggleThinking} 
            />
        ) : (
          <>
            {msg.imagePreviewUrl && (
              <img 
                src={msg.imagePreviewUrl} 
                alt="User upload" 
                style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px', marginBottom: '8px' }} 
              />
            )}
            <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
              {msg.content}
            </ReactMarkdown>
          </>
        )}
      </div>
    </div>
  );
}

export default MessageBubble;