import React from 'react';

function ModelSelector({ models, selectedModel, setSelectedModel, isOverallStreaming }) {
  return (
    <div className="model-selector">
      <select
        id="model"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={isOverallStreaming}
      >
        {models.map(m => {
          const display = m.includes(":") ? m.split(":")[0] : m;
          return <option key={m} value={m}>{display}</option>;
        })}
      </select>
    </div>
  );
}

export default ModelSelector;