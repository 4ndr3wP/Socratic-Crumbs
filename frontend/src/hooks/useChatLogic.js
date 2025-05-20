import { useState, useCallback } from 'react';

// Moved from App.js
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
    responseText = (responseText.substring(0, thinkStartIndex) + responseText.substring(thinkEndIndex + thinkTagEnd.length));
    showToggleButton = true;
  }
  // Trim leading/trailing whitespace and newlines to prevent blank lines
  responseText = responseText.replace(/^\s+/, "").replace(/\s+$/, "");
  return { thinkingText, responseText, showToggleButton };
};

export const useChatLogic = (initialMessages = [], selectedModelProp) => {
  const [messages, setMessages] = useState(initialMessages);
  const [isOverallStreaming, setIsOverallStreaming] = useState(false);
  const [abortController, setAbortController] = useState(null);

  const handleToggleThinking = useCallback((messageId) => {
    setMessages(prevMessages =>
      prevMessages.map(msg =>
        msg.id === messageId ? { ...msg, isThinkingVisible: !msg.isThinkingVisible } : msg
      )
    );
  }, []);

  const handleStopStreaming = useCallback(() => {
    if (abortController) {
      abortController.abort();
      console.log("Streaming stop requested by user.");
    }
  }, [abortController]);

  const handleSubmit = useCallback(async (
    userInput, 
    setUserInput, 
    currentMessages, 
    selectedModel, 
    selectedImage, 
    setSelectedImage, 
    setImagePreview, 
    setAppMessages
  ) => {
    if ((!userInput.trim() && !selectedImage) || isOverallStreaming) return;

    let imageBase64 = null;
    let imagePreviewUrl = null; 

    if (selectedImage) {
      imagePreviewUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(selectedImage);
      });
      imageBase64 = imagePreviewUrl.split(',')[1];
    }

    const newUserMessage = { 
      role: 'user', 
      content: userInput, 
      id: Date.now() + '-user',
      imagePreviewUrl: selectedImage ? imagePreviewUrl : null
    };
    
    setAppMessages(prev => [...prev, newUserMessage]); 
    
    setUserInput(''); 
    if (selectedImage) {
      setSelectedImage(null);
      setImagePreview(null);
    }

    const latestUserMessageForPayload = { 
      role: 'user', 
      content: userInput,
      images: imageBase64 ? [imageBase64] : null 
    };

    const conversationPayload = currentMessages
      .filter(msg => msg.role === 'user' || (msg.role === 'assistant' && msg.isStreamingComplete))
      .map(m => ({ 
        role: m.role, 
        content: m.role === 'assistant' ? m.response : m.content,
        images: (m.role === 'user' && m.id === newUserMessage.id -1 && m.imagePreviewUrl) ? m.images : null
      }))
      .concat([latestUserMessageForPayload]);

    const controller = new AbortController();
    setAbortController(controller);

    const assistantMsgId = Date.now() + '-assistant';

    setAppMessages(prev => [...prev, {
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
        if (controller.signal.aborted) {
          done = true; 
          throw new DOMException('Aborted by user', 'AbortError');
        }
        if (value) {
          const decodedChunk = decoder.decode(value, { stream: true });
          setAppMessages(prevMessages => 
            prevMessages.map(msg =>
              msg.id === assistantMsgId
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
        setAppMessages(prev => prev.map(msg => {
          if (msg.id === assistantMsgId) {
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
        setAppMessages(prev => prev.map(msg => {
          if (msg.id === assistantMsgId) {
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

      setAppMessages(prevMessages =>
        prevMessages.map(msg => {
          if (msg.id === assistantMsgId && !msg.isStreamingComplete) { 
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
  }, [isOverallStreaming]);

  return {
    messages,
    setMessages,
    isOverallStreaming,
    handleToggleThinking,
    handleStopStreaming,
    handleSubmit,
  };
};
