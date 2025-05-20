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
              msg.imagePreviewUrl.isPdf ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  background: '#f6f7fa',
                  borderRadius: '8px',
                  marginBottom: '8px',
                  border: '1px solid #e5e7eb',
                  maxWidth: '80%',
                }}>
                  {/* PDF icon with three lines, matching attachment icon */}
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="8" y1="13" x2="16" y2="13"/>
                    <line x1="8" y1="16" x2="16" y2="16"/>
                    <line x1="8" y1="19" x2="16" y2="19"/>
                  </svg>
                  <span style={{ color: '#2563eb', fontWeight: 500, fontSize: '13px', wordBreak: 'break-all' }}>
                    {msg.imagePreviewUrl.originalName || 'PDF Document'}
                  </span>
                </div>
              ) : (
                <img 
                  src={msg.imagePreviewUrl.url} 
                  alt="User upload" 
                  style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px', marginBottom: '8px' }} 
                />
              )
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