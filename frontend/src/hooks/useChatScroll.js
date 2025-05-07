
import { useState, useEffect, useRef } from 'react';

export const useChatScroll = (dependencies) => {
  const chatContainerRef = useRef(null);
  const [isUserScrolled, setIsUserScrolled] = useState(false);

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
  }, []); // Empty dependency array means this effect runs once on mount and cleanup on unmount

  useEffect(() => {
    const chatDiv = chatContainerRef.current;
    if (chatDiv && !isUserScrolled) {
      chatDiv.scrollTop = chatDiv.scrollHeight;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dependencies, isUserScrolled]); // Auto-scroll when dependencies change, if user hasn't scrolled up

  const handleScrollToBottom = () => {
    const chatDiv = chatContainerRef.current;
    if (chatDiv) {
      chatDiv.scrollTop = chatDiv.scrollHeight;
      setIsUserScrolled(false);
    }
  };

  return {
    chatContainerRef,
    isUserScrolled,
    handleScrollToBottom,
  };
};
