# STS Service (Speech-to-Text-to-Speech)

This module implements a real-time Speech-to-Text-to-Speech (STS) service for conversational AI applications. It is designed to power a voice assistant that listens to user speech, transcribes it, generates a response using an LLM, and speaks the response back to the user.

# Getting Started

git clone the project repository and install the required dependencies.

Open project folder in VS Code or your preferred IDE, and ensure you have the necessary Python environment set up. The service is designed to run as a WebSocket server, allowing real-time interaction with clients.

In terminal, cd into the project directory

"cd frontend && npm run build"

"cd .. && uvicorn main:app --reload"

## Features
- **Real-time voice input** using microphone (with device selection and VAD)
- **Speech-to-text** transcription using Whisper (via MLX)
- **Conversational context** with LLM (Ollama client)
- **Text-to-speech** synthesis with configurable voices and speeds
- **Robust filtering** for echoes, filler words, and system feedback
- **WebSocket API** for interactive frontend integration
- **Session management** for multiple users

## Main Components
- `STSSession`: Handles a single user's audio stream, transcription, LLM response, and TTS playback.
- **Voice Activity Detection (VAD)**: Filters out silence and echoes using energy and VAD checks.
- **Conversation History**: Maintains short context for LLM responses.
- **Audio Player**: Streams synthesized speech to the user.

## Usage
This service is intended to be run as part of a FastAPI backend. The main entrypoint is the `handle_sts_session` coroutine, which manages a WebSocket connection for each user session.

### Example (Python, FastAPI)
```python
from fastapi import WebSocket
from sts_service import handle_sts_session

@app.websocket("/ws/sts")
async def websocket_endpoint(websocket: WebSocket):
    await handle_sts_session(websocket)
```

## Configuration
- **Models**: Selectable LLM and TTS voice per session.
- **Thresholds**: Tunable silence, energy, and timing parameters for robust speech detection.
- **Voice Speeds**: Configurable via `config.py`.

## Requirements
- Python 3.8+
- `mlx.core`, `numpy`, `sounddevice`, `webrtcvad`, `fastapi`, `ollama`, and dependencies in `mlx_audio`.

## File Structure
- `sts_service.py`: Main STS session logic
- `config.py`: Shared configuration (voice speeds, etc.)
- `mlx_audio/`: Audio models and utilities

## Notes
- Designed for macOS, but should work on other platforms with compatible audio devices.
- Handles echo and feedback suppression for natural conversations.

## License
See project root for license information.
