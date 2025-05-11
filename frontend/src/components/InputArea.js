import React, { useEffect, useRef } from 'react'; // Removed useState

// InputArea component
function InputArea({
  userInput,
  setUserInput,
  isOverallStreaming,
  handleSubmit,
  handleStopStreaming,
  selectedImage, // New prop
  setSelectedImage, // New prop
  imagePreview, // New prop
  setImagePreview // New prop
}) {
  const textareaRef = useRef(null); // Ref for the textarea
  const fileInputRef = useRef(null); // Ref for the file input

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

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const onFormSubmit = (e) => {
    e.preventDefault();
    handleSubmit(e); // Pass the event
  };

  return (
    <form className="input-area" onSubmit={onFormSubmit}>
      {imagePreview && (
        <div 
          className="image-preview-container"
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            marginRight: '8px', 
            padding: '2px 4px', // Adjusted padding
            border: '1px solid #ddd', 
            borderRadius: '4px' 
          }}
        >
          <img 
            src={imagePreview} 
            alt="Preview" 
            className="image-preview" 
            style={{ 
              maxHeight: '40px', 
              maxWidth: '60px', 
              borderRadius: '3px', 
              objectFit: 'cover' 
            }}
          />
          <button 
            type="button" 
            onClick={() => { 
              setSelectedImage(null); 
              setImagePreview(null); 
              if(fileInputRef.current) fileInputRef.current.value = null; 
            }} 
            className="remove-image-button"
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              fontSize: '20px', // Slightly larger 'x'
              padding: '0 0 0 6px', // Padding to the left of 'x'
              lineHeight: '1',
              fontWeight: 'bold' // Make 'x' bolder
            }}
          >
            &times;
          </button>
        </div>
      )}
      {!imagePreview && ( // Only show attach button if no image is selected
        <button 
          type="button" 
          onClick={triggerFileInput} 
          className="attach-image-button" 
          disabled={isOverallStreaming} // selectedImage check is implicitly handled by !imagePreview
          style={{ 
            marginRight: '8px', 
            padding: '5px 8px', // Adjusted padding
            border: '1px solid #ccc', 
            borderRadius: '4px', 
            background: '#f0f0f0',
            cursor: 'pointer',
            lineHeight: '1' // Ensure icon aligns well
          }}
        >
          🖼️
        </button>
      )}
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept="image/*"
        onChange={handleImageChange}
        disabled={isOverallStreaming} // selectedImage check is implicitly handled by !imagePreview for the trigger button
      />
      <textarea
        ref={textareaRef} // Add ref to textarea
        value={userInput}
        onChange={handleInputChange}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline in textarea
            if ((userInput.trim() || selectedImage) && e.target.form) { // Allow submit if image is selected
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
        <button type="submit" disabled={!(userInput.trim() || selectedImage)}>Send</button> // Allow submit if image is selected
      ) : (
        <button type="button" onClick={handleStopStreaming}>Stop</button>
      )}
    </form>
  );
}

export default InputArea;