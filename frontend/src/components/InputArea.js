import React, { useEffect, useRef } from 'react';

// InputArea component
function InputArea({
  userInput,
  setUserInput,
  isOverallStreaming,
  handleSubmit,
  handleStopStreaming
}) {
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

  return (
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
  );
}

export default InputArea;