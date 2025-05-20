import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

// AssistantBubble component
function AssistantBubble({ message, onToggleThinking }) {
  const { 
    content, 
    thinking, 
    response, 
    isStreamingComplete, 
    showToggle 
  } = message;

  const [isTooltipVisible, setIsTooltipVisible] = React.useState(false);

  const handleCopy = (code) => {
    navigator.clipboard.writeText(code);
  };

  // Always use the same markdown rendering logic for both streaming and complete states
  const renderMarkdown = (text) => (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const lang = match ? match[1].toLowerCase() : null;

          if (inline || lang === 'latex' || lang === 'markdown') {
            if (inline) {
              return <code className={className} {...props}>{children}</code>;
            } else {
              return <pre><code className={className} {...props}>{String(children).replace(/\n$/, '')}</code></pre>;
            }
          }
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
          return <pre><code className={className} {...props}>{String(children).replace(/\n$/, '')}</code></pre>;
        }
      }}
    >
      {text ? text.replace(/^\s+/, "") : ""}
    </ReactMarkdown>
  );

  if (!isStreamingComplete && content) {
    return renderMarkdown(content);
  }
  if (isStreamingComplete) {
    return (
      <div className="assistant-bubble-content">
        {showToggle && (
          <div
            className="think-icon-area"
            onMouseEnter={() => setIsTooltipVisible(true)}
            onMouseLeave={() => setIsTooltipVisible(false)}
          >
            <div className="think-toggle-container">
              <button className="think-toggle-button" aria-label="Show Thoughts">
                <img src="/SocraticCrumbsIcon.png" alt="Thinking process" />
              </button>
            </div>
            {isTooltipVisible && thinking && (
              <div className="thinking-tooltip">
                {renderMarkdown(thinking)}
              </div>
            )}
          </div>
        )}
        {renderMarkdown(response)}
      </div>
    );
  }
  return null;
}

export default AssistantBubble;