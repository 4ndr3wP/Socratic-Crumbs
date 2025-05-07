import React from 'react';

function ModelSelector({ models, selectedModel, setSelectedModel, isOverallStreaming }) {
  return (
    <div className="model-selector">
      <label htmlFor="model">Model:</label>
      <select
        id="model"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={isOverallStreaming}
      >
        {models.map(m => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </div>
  );
}

export default ModelSelector;