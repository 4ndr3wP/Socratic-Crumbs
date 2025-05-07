import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

// AssistantBubble component
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

              // Handle inline code, or specific block languages like 'latex' (for KaTeX) 
              // and 'markdown' (to prevent syntax highlighting of Markdown content itself).
              if (inline || lang === 'latex' || lang === 'markdown') {
                if (inline) {
                  // Standard inline code rendering
                  return <code className={className} {...props}>{children}</code>;
                } else {
                  // For block-level 'latex' or 'markdown', render as simple preformatted text.
                  // This avoids applying the SyntaxHighlighter's theme (e.g., black box) to them.
                  return <pre><code className={className} {...props}>{String(children).replace(/\n$/, '')}</code></pre>;
                }
              }
              
              // If it's a block code with a recognized language for syntax highlighting (and not handled above)
              if (lang) { 
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
              
              // Fallback for block code without a specified language or any other unhandled case.
              // Render as simple preformatted text.
              return <pre><code className={className} {...props}>{String(children).replace(/\n$/, '')}</code></pre>;
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

export default AssistantBubble;