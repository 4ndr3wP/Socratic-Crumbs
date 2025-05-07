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
    responseText = (responseText.substring(0, thinkStartIndex) + responseText.substring(thinkEndIndex + thinkTagEnd.length)).trim();
    showToggleButton = true;
  }
  return { thinkingText, responseText, showToggleButton };
};

export const useChatLogic = (initialMessages = [], selectedModelProp) => {
  const [messages, setMessages] = useState(initialMessages);
  const [isOverallStreaming, setIsOverallStreaming] = useState(false);
  const [abortController, setAbortController] = useState(null);
  // selectedModel is managed by App.js, passed as a prop to the hook
  // const [selectedModel, setSelectedModel] = useState(selectedModelProp); 
  // useEffect(() => { setSelectedModel(selectedModelProp); }, [selectedModelProp]);


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

  const handleSubmit = useCallback(async (userInput, setUserInput, currentMessages, selectedModel) => {
    if (!userInput.trim() || isOverallStreaming) return;

    const newUserMessage = { role: 'user', content: userInput, id: Date.now() + '-user' };
    setMessages(prev => [...prev, newUserMessage]);
    setUserInput(''); // Clear input in App.js via callback

    const conversationPayload = currentMessages // Use currentMessages from App.js state at time of call
      .filter(msg => msg.role === 'user' || (msg.role === 'assistant' && msg.isStreamingComplete))
      .map(m => ({ role: m.role, content: m.role === 'assistant' ? m.response : m.content }))
      .concat([{ role: 'user', content: userInput }]);

    const controller = new AbortController();
    setAbortController(controller);

    const assistantMsgId = Date.now() + '-assistant';

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
        if (controller.signal.aborted) { // Check for abort signal
          done = true; // Exit loop if aborted
          throw new DOMException('Aborted by user', 'AbortError');
        }
        if (value) {
          const decodedChunk = decoder.decode(value, { stream: true });
          setMessages(prevMessages =>
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
        setMessages(prev => prev.map(msg => {
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
        setMessages(prev => prev.map(msg => {
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

      setMessages(prevMessages =>
        prevMessages.map(msg => {
          if (msg.id === assistantMsgId && !msg.isStreamingComplete) { // Ensure this runs only if not already completed by abort
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
  }, [isOverallStreaming]); // Dependencies: selectedModel will be passed directly

  return {
    messages,
    setMessages, // Expose setMessages for initialization if needed
    isOverallStreaming,
    handleToggleThinking,
    handleStopStreaming,
    handleSubmit,
  };
};
