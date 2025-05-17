import { useState, useEffect } from 'react';

export const useModels = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('lastSelectedModel') || '');

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        if (data.models) {
          setModels(data.models);
          const lastSelected = localStorage.getItem('lastSelectedModel');
          if (data.models.length > 0) {
            if (lastSelected && data.models.includes(lastSelected)) {
              setSelectedModel(lastSelected);
            } else if (!selectedModel) {
              setSelectedModel(data.models[0]);
            }
          }
        }
      })
      .catch(err => console.error('Failed to load models:', err));
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
