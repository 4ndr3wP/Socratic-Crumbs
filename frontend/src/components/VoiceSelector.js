import React from 'react';

function VoiceSelector({ voices, selectedVoice, setSelectedVoice, isStreaming }) {
  const handleVoiceChange = (e) => {
    setSelectedVoice(e.target.value);
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
            {voice.replace('af_', '').replace('bf_', '').replace('.pt', '')}
          </option>
        ))}
      </select>
    </div>
  );
}

export default VoiceSelector; 