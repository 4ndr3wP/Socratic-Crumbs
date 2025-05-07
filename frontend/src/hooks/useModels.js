import { useState, useEffect } from 'react';

export const useModels = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setModels(data.models);
          if (data.models.length > 0 && !selectedModel) { // Set initial model only if not already set
            setSelectedModel(data.models[0]);
          }
        }
      })
      .catch(err => console.error('Failed to load models:', err));
  // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, []); // Fetch models only once on mount, selectedModel dependency removed as it's set internally.

  return {
    models,
    selectedModel,
    setSelectedModel,
  };
};
