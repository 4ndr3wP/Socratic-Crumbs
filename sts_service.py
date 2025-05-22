import asyncio
import logging
import os
import uuid
import json
import time
from typing import Dict, Optional
from ollama import AsyncClient

import mlx.core as mx
import numpy as np
import sounddevice as sd
import webrtcvad
from fastapi import WebSocket

from mlx_audio.stt.models.whisper import Model as Whisper
from mlx_audio.stt.generate import generate as stt_generate
from mlx_audio.tts.audio_player import AudioPlayer
from mlx_audio.tts.generate import generate_audio

# Import voice speeds from shared config
from config import VOICE_SPEEDS

logger = logging.getLogger("sts_service")
logger.setLevel(logging.INFO)

# Configuration for STS - these are just defaults that will be overridden
STS_CONFIG = {
    "silence_threshold": 0.03,
    "silence_duration": 1.0,
    "input_sample_rate": 16000,
    "output_sample_rate": 24000,
    "streaming_interval": 3,
    "frame_duration_ms": 30,
    "vad_mode": 3,
    # Models will be dynamically set based on user selection
    "stt_model": None,      # Will use the STT model from main.py
    "llm_model": None,      # Will use the LLM selected by the user
    "voice": None,          # Will use the voice selected by the user
}

# Dictionary to store active STS sessions
active_sessions: Dict[str, "STSSession"] = {}

class STSSession:
    def __init__(
        self,
        session_id: str,
        websocket: WebSocket,
        llm_model: str,       # LLM model selected by the user
        voice: str,           # Voice selected by the user
        config: dict = None,
    ):
        config = config or STS_CONFIG
        self.session_id = session_id
        self.websocket = websocket
        self.silence_threshold = config.get("silence_threshold", 0.03)
        self.silence_duration = config.get("silence_duration", 1.0)
        self.input_sample_rate = config.get("input_sample_rate", 16000)
        self.output_sample_rate = config.get("output_sample_rate", 24000)
        self.streaming_interval = config.get("streaming_interval", 3)
        self.frame_duration_ms = config.get("frame_duration_ms", 30)
        self.vad_mode = config.get("vad_mode", 3)
        
        # Use the provided models from the user selection
        self.stt_model = "mlx-community/whisper-large-v3-turbo"  # Using the standard STT model
        self.llm_model = llm_model  # Use the model selected by the user
        self.voice = voice  # Use the voice selected by the user
        
        self.vad = webrtcvad.Vad(self.vad_mode)
        
        # Queues for audio processing
        self.input_audio_queue = asyncio.Queue(maxsize=50)
        self.transcription_queue = asyncio.Queue()
        self.output_audio_queue = asyncio.Queue(maxsize=50)
        
        # Lock for MLX operations
        self.mlx_lock = asyncio.Lock()
        
        # Audio state
        self.frames = []
        self.silent_frames = 0
        self.speaking_detected = False
        self.frames_until_silence = int(self.silence_duration * 1000 / self.frame_duration_ms)
        self.stream = None
        self.loop = None
        self.is_active = False
        self.current_tts_cancel = None
        self.current_tts_task = None
        
        # Models and clients
        self.stt = None
        self.ollama_client = None
        self.player = None
        
        logger.info(f"Created new STS session: {session_id} with LLM: {llm_model}, Voice: {voice}")
    
    async def init_models(self):
        # Initialize STT model
        logger.info(f"Loading speech-to-text model: {self.stt_model}")
        self.stt = Whisper.from_pretrained(self.stt_model)
        
        # Initialize Ollama client for LLM
        logger.info(f"Initializing Ollama client for model: {self.llm_model}")
        self.ollama_client = AsyncClient()
        
        # No need to explicitly initialize TTS - we'll use generate_audio directly
        # when needed in the _speak_response method
        
        logger.info(f"All models loaded for session {self.session_id}")
    
    async def start(self):
        if self.is_active:
            return
            
        self.loop = asyncio.get_running_loop()
        self.is_active = True
        
        await self.init_models()
        
        self.tasks = [
            asyncio.create_task(self._listener()),
            asyncio.create_task(self._response_processor()),
            asyncio.create_task(self._audio_output_processor()),
        ]
        
        logger.info(f"Started STS session: {self.session_id}")
        await self.websocket.send_json({"status": "ready", "message": "STS is now active and listening"})
    
    async def stop(self):
        if not self.is_active:
            return
            
        self.is_active = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.player:
            self.player.stop()
            
        logger.info(f"Stopped STS session: {self.session_id}")
        
        # Try to send final message if websocket is still open
        try:
            await self.websocket.send_json({"status": "stopped", "message": "STS session ended"})
        except Exception:
            pass
    
    def _is_silent(self, audio_data):
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0
        else:
            audio_np = audio_data

        # Ensure audio_np is float32 for energy calculation
        audio_np = audio_np.astype(np.float32)

        energy = np.linalg.norm(audio_np) / np.sqrt(audio_np.size)
        return energy < self.silence_threshold
    
    def _voice_activity_detection(self, frame):
        try:
            return self.vad.is_speech(frame, self.input_sample_rate)
        except ValueError:
            # Fall back to energy-based detection
            return not self._is_silent(frame)
    
    def _sd_callback(self, indata, frames, _time, status):
        if not self.is_active:
            return
            
        data = indata.reshape(-1).tobytes()

        def _enqueue():
            try:
                self.input_audio_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # Drop frame if queue is full

        self.loop.call_soon_threadsafe(_enqueue)
    
    async def _listener(self):
        frame_size = int(self.input_sample_rate * (self.frame_duration_ms / 1000.0))
        self.stream = sd.InputStream(
            samplerate=self.input_sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="int16",
            callback=self._sd_callback,
        )
        self.stream.start()

        logger.info(f"Session {self.session_id} listening for voice input...")
        await self.websocket.send_json({"status": "listening", "message": "Listening for voice input..."})

        try:
            while self.is_active:
                try:
                    frame = await asyncio.wait_for(self.input_audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                    
                is_speech = self._voice_activity_detection(frame)

                if is_speech:
                    if not self.speaking_detected:
                        await self.websocket.send_json({"status": "speech_detected", "message": "Speech detected"})
                        
                    self.speaking_detected = True
                    self.silent_frames = 0
                    self.frames.append(frame)
                elif self.speaking_detected:
                    self.silent_frames += 1
                    self.frames.append(frame)

                    if self.silent_frames > self.frames_until_silence:
                        # Process the voice input
                        if self.frames:
                            logger.info(f"Session {self.session_id} processing voice input...")
                            await self.websocket.send_json({"status": "processing", "message": "Processing speech..."})
                            await self._process_audio(self.frames)

                        self.frames = []
                        self.speaking_detected = False
                        self.silent_frames = 0
        except asyncio.CancelledError:
            logger.info(f"Listener task cancelled for session {self.session_id}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    async def _process_audio(self, frames):
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        text = result.text.strip()

        if text:
            logger.info(f"Session {self.session_id} transcribed: {text}")
            await self.websocket.send_json({"status": "transcribed", "message": f"Transcribed: {text}"})
            await self.transcription_queue.put(text)
    
    async def _response_processor(self):
        try:
            while self.is_active:
                try:
                    text = await asyncio.wait_for(self.transcription_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                    
                await self._generate_response(text)
                self.transcription_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Response processor task cancelled for session {self.session_id}")
    
    async def _generate_response(self, text):
        try:
            logger.info(f"Session {self.session_id} generating response...")
            await self.websocket.send_json({"status": "generating", "message": "Generating response..."})

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. You always respond with short sentences and never use punctuation like parentheses or colons that wouldn't appear in conversational speech.",
                },
                {"role": "user", "content": text},
            ]
            
            # Generate response using Ollama
            response = await self.ollama_client.chat(
                model=self.llm_model,
                messages=messages,
                stream=False
            )
            
            response_text = response["message"]["content"].strip()

            logger.info(f"Session {self.session_id} generated response: {response_text}")
            await self.websocket.send_json({"status": "response", "message": f"Response: {response_text}"})

            if response_text:
                self.current_tts_cancel = asyncio.Event()
                self.current_tts_task = asyncio.create_task(
                    self._speak_response(response_text, self.current_tts_cancel)
                )
        except Exception as e:
            logger.error(f"Session {self.session_id} generation error: {e}")
            await self.websocket.send_json({"status": "error", "message": f"Error generating response: {str(e)}"})
    
    async def _speak_response(self, text: str, cancel_event: asyncio.Event):
        """
        Speak `text` using the voice selected by the user.
        Uses the existing TTS functionality in the main app.
        """
        try:
            await self.websocket.send_json({"status": "speaking", "message": "Speaking response..."})
            
            # Get the speed for the selected voice
            speed = VOICE_SPEEDS.get(self.voice, 1.0)
            
            loop = asyncio.get_running_loop()
            
            # Create an audio queue for streaming output
            audio_queue = asyncio.Queue()
            
            # Initialize player if not already done
            if not self.player:
                self.player = AudioPlayer(sample_rate=self.output_sample_rate)
            
            # Run TTS generation in a separate thread to avoid blocking
            def _tts_stream():
                try:
                    # Generate audio segments
                    for segment in generate_audio(
                        text=text,
                        voice=self.voice,
                        speed=speed,
                        lang_code=self.voice[0],  # First character of voice name is lang code
                        verbose=False,
                        stream=True,
                        streaming_interval=self.streaming_interval
                    ):
                        if cancel_event and cancel_event.is_set():
                            break
                        # Send audio data to the queue
                        loop.call_soon_threadsafe(audio_queue.put_nowait, segment.audio)
                except Exception as e:
                    logger.error(f"TTS generation error: {e}")
                finally:
                    # Signal the end of streaming
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)
            
            # Start TTS generation in a thread
            tts_thread = asyncio.create_task(asyncio.to_thread(_tts_stream))
            
            # Process audio chunks as they become available
            while True:
                try:
                    # Wait for the next audio chunk
                    audio_chunk = await audio_queue.get()
                    
                    # None signals the end of streaming
                    if audio_chunk is None:
                        break
                        
                    # Add to our output queue for the audio player
                    await self.output_audio_queue.put(audio_chunk)
                    
                    # Check if we should cancel
                    if cancel_event and cancel_event.is_set():
                        break
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    break
            
            # Make sure the TTS thread is done
            await tts_thread
            
            await self.websocket.send_json({"status": "listening", "message": "Listening for voice input..."})
            
        except asyncio.CancelledError:
            # The coroutine itself was cancelled from outside → just exit cleanly.
            pass
        except Exception as exc:
            logger.error(f"Session {self.session_id} speech synthesis error: {exc}")
            await self.websocket.send_json({"status": "error", "message": f"Error synthesizing speech: {str(exc)}"})
            if cancel_event:
                cancel_event.set()
    
    async def _audio_output_processor(self):
        self.player = AudioPlayer(sample_rate=self.output_sample_rate)

        try:
            while self.is_active:
                try:
                    audio = await asyncio.wait_for(self.output_audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                    
                self.player.queue_audio(audio)
                self.output_audio_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Audio output processor task cancelled for session {self.session_id}")
            if self.player:
                self.player.stop()
        finally:
            if self.player:
                self.player.stop()

async def handle_sts_session(websocket: WebSocket, session_id: str = None, llm_model: str = None, voice: str = None):
    """
    Handle a STS session over WebSocket
    
    Parameters:
    - websocket: The WebSocket connection
    - session_id: Optional session ID (will be generated if not provided)
    - llm_model: The LLM model to use (should be selected by the user)
    - voice: The voice to use for TTS (should be selected by the user)
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # Default values if not provided
    if not llm_model:
        llm_model = "llama3"  # Default model
    if not voice:
        voice = "af_heart"    # Default voice
        
    await websocket.accept()
    
    try:
        # Create a new STS session with the user-selected models
        session = STSSession(
            session_id=session_id,
            websocket=websocket,
            llm_model=llm_model,
            voice=voice
        )
        active_sessions[session_id] = session
        
        # Start the session
        await session.start()
        
        # Keep the WebSocket connection open
        while True:
            try:
                data = await websocket.receive_text()
                msg = {"status": "received", "message": f"Received command: {data}"}
                
                if data == "stop":
                    await session.stop()
                    del active_sessions[session_id]
                    break
                elif data.startswith("config:"):
                    # Handle configuration changes
                    config_str = data[7:]
                    try:
                        config = json.loads(config_str)
                        # Update voice if provided
                        if "voice" in config:
                            session.voice = config["voice"]
                            msg = {"status": "config", "message": f"Voice updated to {session.voice}"}
                        # Update model if provided
                        if "model" in config:
                            session.llm_model = config["model"]
                            msg = {"status": "config", "message": f"Model updated to {session.llm_model}"}
                    except Exception as e:
                        msg = {"status": "error", "message": f"Invalid configuration: {str(e)}"}
                
                await websocket.send_json(msg)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    break
                logger.error(f"WebSocket error: {str(e)}")
                break
    except Exception as e:
        logger.error(f"Error in STS session {session_id}: {str(e)}")
    finally:
        # Clean up the session
        if session_id in active_sessions:
            await active_sessions[session_id].stop()
            del active_sessions[session_id]
        
        try:
            await websocket.close()
        except Exception:
            pass
