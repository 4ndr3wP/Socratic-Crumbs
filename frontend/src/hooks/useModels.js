import { useState, useEffect } from 'react';

export const useModels = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('lastSelectedModel') || '');

  useEffect(() => {
    // Fetch model tags from Ollama API to get parameter sizes
    fetch('http://localhost:11434/api/tags')
      .then(res => res.json())
      .then(tagData => {
        if (tagData.models && Array.isArray(tagData.models)) {
          // Map model name to size
          const modelSizes = {};
          tagData.models.forEach(m => {
            if (m.name && m.size) {
              modelSizes[m.name] = m.size;
            }
          });
          // Now fetch the models from the backend as before
          fetch('/api/models')
            .then(res => res.json())
            .then(data => {
              if (data.models) {
                // Sort models by size from Ollama API (descending)
                const sortedModels = [...data.models].sort((b, a) => {
                  const sizeA = modelSizes[a] || 0;
                  const sizeB = modelSizes[b] || 0;
                  return sizeB - sizeA;
                });
                setModels(sortedModels);
                const lastSelected = localStorage.getItem('lastSelectedModel');
                if (sortedModels.length > 0) {
                  if (lastSelected && sortedModels.includes(lastSelected)) {
                    setSelectedModel(lastSelected);
                  } else if (!selectedModel) {
                    setSelectedModel(sortedModels[0]);
                  }
                }
              }
            });
        }
      })
      .catch(err => console.error('Failed to load model sizes or models:', err));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (selectedModel) {
      localStorage.setItem('lastSelectedModel', selectedModel);
    }
  }, [selectedModel]);

  return {
    models,
    selectedModel,
    setSelectedModel,
  };
};
