import React, { useEffect } from 'react';

function VoiceSelector({ voices, selectedVoice, setSelectedVoice, isStreaming }) {
  const handleVoiceChange = (e) => {
    const newVoice = e.target.value;
    setSelectedVoice(newVoice);
    // Save the selected voice to localStorage
    localStorage.setItem('lastSelectedVoice', newVoice);
  };

  // Load the last selected voice on component mount
  useEffect(() => {
    const lastVoice = localStorage.getItem('lastSelectedVoice');
    if (lastVoice && voices.includes(lastVoice)) {
      setSelectedVoice(lastVoice);
    }
  }, [voices, setSelectedVoice]);

  const formatVoiceName = (voice) => {
    // Remove prefixes and capitalize each word
    return voice
      .replace('af_', '')
      .replace('bf_', '')
      .replace('im_', '')
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="voice-selector">
      <select
        value={selectedVoice}
        onChange={handleVoiceChange}
        disabled={isStreaming}
        style={{
          padding: '4px 8px',
          borderRadius: '4px',
          border: '1px solid #ccc',
          backgroundColor: '#fff',
          fontSize: '14px',
          cursor: isStreaming ? 'not-allowed' : 'pointer',
          opacity: isStreaming ? 0.7 : 1
        }}
      >
        {voices.map((voice) => (
          <option key={voice} value={voice}>
            {formatVoiceName(voice)}
          </option>
        ))}
      </select>
    </div>
  );
}

export default VoiceSelector; 