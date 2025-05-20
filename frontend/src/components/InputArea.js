import React, { useEffect, useRef } from 'react'; // Removed useState
import * as pdfjsLib from 'pdfjs-dist';

// Set up PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

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
  setImagePreview, // New prop
  isAudioPlaying,
  isTTSLoading
}) {
  const textareaRef = useRef(null); // Ref for the textarea
  const fileInputRef = useRef(null); // Ref for the file input
  const canvasRef = useRef(null);

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

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      if (file.type === 'application/pdf') {
        // Extract text from PDF
        let extractedText = '';
        try {
          const arrayBuffer = await file.arrayBuffer();
          const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
          const numPages = pdf.numPages;
          for (let i = 1; i <= numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items.map(item => item.str).join(' ');
            extractedText += pageText + '\n';
          }
        } catch (err) {
          console.error('Failed to extract PDF text:', err);
        }
        setImagePreview({
          isPdf: true,
          originalName: file.name,
          pdfText: extractedText
        });
      } else {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview({
            url: reader.result,
            isPdf: false
          });
        };
        reader.readAsDataURL(file);
      }
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
    if (isOverallStreaming || isAudioPlaying) {
      handleStopStreaming();
    } else {
      handleSubmit(e);
    }
  };

  const isBusy = isOverallStreaming || isAudioPlaying || isTTSLoading;
  const hasText = userInput.trim().length > 0;

  // Button color logic
  let buttonBg = '#f3f4f6'; // default grey
  let buttonColor = '#9ca3af'; // default grey text
  let buttonBorder = 'none';
  if (isBusy) {
    buttonBg = '#fff'; // white for Stop
    buttonColor = '#333'; // dark text
    buttonBorder = '1px solid #ddd'; // subtle border
  } else if (hasText) {
    buttonBg = '#2563eb'; // blue-600 (matches prompt bubble)
    buttonColor = 'white';
    buttonBorder = 'none';
  }

  return (
    <form onSubmit={onFormSubmit} className="input-area" style={{ width: '100%', padding: '16px', background: '#fff', borderTop: '1px solid #eee' }}>
      <div className="input-container" style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
        {/* Attachment icon */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', height: '36px' }}>
          <button
            type="button"
            onClick={selectedImage ? () => { setSelectedImage(null); setImagePreview(null); if(fileInputRef.current) fileInputRef.current.value = null; } : triggerFileInput}
            className="attach-image-button"
            disabled={isBusy}
            style={{
              background: 'none',
              border: 'none',
              padding: 0,
              marginRight: '4px',
              cursor: isBusy ? 'not-allowed' : 'pointer',
              opacity: isBusy ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              height: '36px',
              color: selectedImage ? '#2563eb' : '#333',
              position: 'relative',
              zIndex: 1
            }}
            title={selectedImage ? 'Remove attachment' : 'Attach image'}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M16 13H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M16 17H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M10 9H9H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            {selectedImage && (
              <span
                style={{
                  position: 'absolute',
                  top: '-4px',
                  right: '-4px',
                  background: '#f3f4f6',
                  color: '#888',
                  borderRadius: '50%',
                  width: '16px',
                  height: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '13px',
                  fontWeight: 700,
                  boxShadow: '0 1px 2px rgba(0,0,0,0.07)',
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  zIndex: 2
                }}
                onClick={e => {
                  e.stopPropagation();
                  setSelectedImage(null);
                  setImagePreview(null);
                  if(fileInputRef.current) fileInputRef.current.value = null;
                }}
                title="Remove attachment"
              >
                ×
              </span>
            )}
          </button>
        </div>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*,.pdf"
          onChange={handleImageChange}
          disabled={isBusy}
        />
        {/* Text box */}
        <textarea
          ref={textareaRef}
          value={userInput}
          onChange={handleInputChange}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if ((userInput.trim() || selectedImage) && e.target.form) {
                e.target.form.requestSubmit();
              }
            }
          }}
          placeholder="Type your message..."
          disabled={isBusy}
          rows={1}
          style={{
            resize: 'none',
            width: '100%',
            minHeight: '36px',
            maxHeight: '120px',
            padding: '8px 12px',
            borderRadius: '8px',
            border: '1px solid #e5e7eb',
            backgroundColor: '#fff',
            fontSize: '15px',
            lineHeight: '1.5',
            color: '#1f2937',
            outline: 'none',
            transition: 'border-color 0.2s',
            opacity: isBusy ? 0.7 : 1,
            cursor: isBusy ? 'not-allowed' : 'text',
            marginRight: '8px',
            flex: 1
          }}
        />
        {/* Send/Stop button */}
        <button
          type="submit"
          disabled={isBusy ? false : !hasText}
          style={{
            padding: '8px 22px',
            backgroundColor: buttonBg,
            color: buttonColor,
            border: buttonBorder,
            borderRadius: '7px',
            cursor: isBusy ? 'pointer' : (!hasText ? 'not-allowed' : 'pointer'),
            fontSize: '15px',
            fontWeight: 600,
            transition: 'all 0.2s',
            boxShadow: '0 1px 2px rgba(0,0,0,0.03)'
          }}
        >
          {isBusy ? 'Stop' : 'Send'}
        </button>
      </div>
    </form>
  );
}

export default InputArea;