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
    "silence_threshold": 0.02,  # Lower threshold to be more sensitive to speech
    "silence_duration": 2.0,    # Increased from 1.0 to 2.0 seconds
    "input_sample_rate": 16000,
    "output_sample_rate": 24000,
    "streaming_interval": 3,
    "frame_duration_ms": 30,
    "vad_mode": 2,              # Less aggressive VAD mode
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
        # Log the default input device at session creation
        try:
            default_input_index = sd.default.device[0]
            input_device_info = sd.query_devices(default_input_index)
            logger.info(f"[{session_id}] Default input device index: {default_input_index}, name: {input_device_info['name']}")
        except Exception as e:
            logger.warning(f"[{session_id}] Could not query default input device: {e}")
        logger.info(f"[{session_id}] Initializing STS session with model={llm_model}, voice={voice}")
        logger.info(f"[{session_id}] Configuration: silence_threshold={config.get('silence_threshold')}, silence_duration={config.get('silence_duration')}")
        self.silence_threshold = config.get("silence_threshold", 0.015)  # Adjusted threshold to work with actual microphone levels
        self.silence_duration = config.get("silence_duration", 1.0)
        self.input_sample_rate = config.get("input_sample_rate", 16000)
        self.output_sample_rate = config.get("output_sample_rate", 24000)
        self.streaming_interval = config.get("streaming_interval", 3)
        self.frame_duration_ms = config.get("frame_duration_ms", 30)
        self.vad_mode = config.get("vad_mode", 2)  # Less aggressive VAD mode
        
        # Use the provided models from the user selection
        self.stt_model = "mlx-community/whisper-large-v3-turbo"
        self.llm_model = llm_model
        self.voice = voice
        
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
        
        # System is speaking flag and cooldown timer
        self.system_is_speaking = False
        self.speech_end_time = 0
        
        # Minimum speech requirements - adjusted for better sensitivity
        self.min_speech_frames = int(0.8 * 1000 / self.frame_duration_ms)  # Reduced to 0.8 seconds minimum
        self.min_speech_energy = 0.010  # Lowered threshold to accommodate actual speech levels
        
        # Simple echo detection - only reject obvious echoes
        self.echo_similarity_threshold = 0.8  # High threshold to only catch clear echoes
        
        # Models and clients
        self.stt = None
        self.ollama_client = None
        self.player = None
        
        # Conversation history
        self.conversation_history = []
        
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
            logger.warning(f"[{self.session_id}] Attempted to start already active session")
            return
            
        logger.info(f"[{self.session_id}] Starting STS session")
        self.loop = asyncio.get_running_loop()
        self.is_active = True
        
        try:
            await self.init_models()
            logger.info(f"[{self.session_id}] Models initialized successfully")
            
            self.tasks = [
                asyncio.create_task(self._listener()),
                asyncio.create_task(self._response_processor()),
                asyncio.create_task(self._audio_output_processor()),
            ]
            logger.info(f"[{self.session_id}] All tasks created and started")
            
            await self.websocket.send_json({"status": "ready", "message": "STS is now active and listening"})
            logger.info(f"[{self.session_id}] Session fully started and ready")
        except Exception as e:
            logger.error(f"[{self.session_id}] Error during session start: {str(e)}", exc_info=True)
            self.is_active = False
            raise
    
    async def stop(self):
        logger.info(f"[{self.session_id}] Stop called. Current state: is_active={self.is_active}")
        if not self.is_active:
            logger.info(f"[{self.session_id}] Session already stopped")
            return
            
        self.is_active = False
        logger.info(f"[{self.session_id}] Stopping session...")
        
        # Cancel current TTS task immediately to stop audio generation
        if self.current_tts_cancel:
            self.current_tts_cancel.set()
        if self.current_tts_task:
            try:
                self.current_tts_task.cancel()
                await asyncio.sleep(0.1)  # Give task time to cancel
            except Exception as e:
                logger.debug(f"[{self.session_id}] Error cancelling TTS task: {e}")
        
        # Immediately flush audio player to stop all playback
        if self.player:
            try:
                logger.info(f"[{self.session_id}] Flushing audio player to stop playback immediately")
                self.player.flush()
                logger.info(f"[{self.session_id}] Audio player flushed successfully")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error flushing audio player: {str(e)}")
        
        # Clear output audio queue
        while not self.output_audio_queue.empty():
            try:
                self.output_audio_queue.get_nowait()
                self.output_audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Cancel all tasks
        for task in getattr(self, 'tasks', []):
            try:
                task.cancel()
                logger.debug(f"[{self.session_id}] Task cancelled: {task.get_name()}")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error cancelling task: {str(e)}")
        
        # Close audio stream
        if self.stream:
            try:
                logger.info(f"[{self.session_id}] Stopping audio input stream")
                self.stream.stop()
                self.stream.close()
                logger.info(f"[{self.session_id}] Audio input stream closed")
            except Exception as e:
                logger.error(f"[{self.session_id}] Error closing audio stream: {str(e)}")
        
        # Clear conversation history
        self.conversation_history = []
        logger.info(f"[{self.session_id}] Conversation history cleared")
        
        # Reset speech state
        self.system_is_speaking = False
            
        logger.info(f"[{self.session_id}] Session fully stopped")
        
        # Try to send final message if websocket is still open
        try:
            if self.websocket.client_state.value == 1:  # WebSocket is still open
                await self.websocket.send_json({"status": "stopped", "message": "STS session ended"})
                logger.info(f"[{self.session_id}] Stop confirmation sent to client")
        except Exception as e:
            logger.debug(f"[{self.session_id}] WebSocket already closed: {e}")
    
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
            # More robust VAD - combine energy check with VAD
            if isinstance(frame, bytes):
                audio_np = np.frombuffer(frame, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                audio_np = frame

            # Calculate energy of the frame
            energy = np.linalg.norm(audio_np) / np.sqrt(audio_np.size)
            
            # First check: energy must be above silence threshold
            if energy < self.silence_threshold:
                logger.debug(f"Session {self.session_id} energy too low: {energy:.4f} < {self.silence_threshold}")
                return False
                
            # Use VAD for speech detection, but be less strict about energy
            try:
                is_speech = self.vad.is_speech(frame, self.input_sample_rate)
                logger.debug(f"Session {self.session_id} VAD result: energy={energy:.4f}, is_speech={is_speech}")
                return is_speech
            except ValueError as e:
                logger.warning(f"Session {self.session_id} VAD error, using energy fallback: {str(e)}")
                # If VAD fails, fall back to energy-based detection with a reasonable threshold
                return energy > self.silence_threshold * 1.5
            
        except Exception as e:
            logger.warning(f"Session {self.session_id} VAD error: {str(e)}")
            return False
    
    def _sd_callback(self, indata, frames, _time, status):
        if not self.is_active:
            return
            
        # Only skip if system is currently speaking, not during cooldown
        if self.system_is_speaking:
            logger.debug(f"Session {self.session_id} skipping audio frame while system is speaking")
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
        # Log all available devices and select the MacBook's built-in microphone
        try:
            logger.info(f"Session {self.session_id} available audio devices:")
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                logger.info(f"  Device {idx}: {dev['name']} (max input channels: {dev['max_input_channels']}, max output channels: {dev['max_output_channels']})")
            
            # Prioritize MacBook's built-in microphone over external devices
            builtin_input_index = None
            fallback_input_index = None
            
            for idx, dev in enumerate(devices):
                name = dev['name'].lower()
                if dev['max_input_channels'] > 0:
                    # Look specifically for MacBook built-in microphone
                    if any(x in name for x in ["built-in", "macbook", "internal"]):
                        builtin_input_index = idx
                        logger.info(f"Session {self.session_id} found MacBook built-in microphone: {dev['name']} (index {idx})")
                        break
                    # Avoid external devices like iPhone, AirPods, etc.
                    elif not any(x in name for x in ["iphone", "airpods", "bluetooth", "virtual", "loopback", "blackhole", "monitor", "output", "speaker"]):
                        if fallback_input_index is None:
                            fallback_input_index = idx
            
            # Use built-in microphone if found, otherwise use fallback
            selected_input_index = builtin_input_index if builtin_input_index is not None else fallback_input_index
            
            if selected_input_index is None:
                logger.warning(f"Session {self.session_id} could not find a suitable input device, using default.")
                selected_input_index = sd.default.device[0]
            else:
                # Set this as the default input device
                sd.default.device = (selected_input_index, sd.default.device[1])
            
            input_device_info = sd.query_devices(selected_input_index)
            logger.info(f"Session {self.session_id} using input device index: {selected_input_index}, name: {input_device_info['name']}")
        except Exception as e:
            logger.warning(f"Session {self.session_id} Could not query audio devices: {e}")
            selected_input_index = None
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                blocksize=frame_size,
                channels=1,
                dtype="int16",
                callback=self._sd_callback,
                device=selected_input_index if selected_input_index is not None else None,
            )
            logger.info(f"Session {self.session_id} successfully created audio input stream")
            self.stream.start()
            logger.info(f"Session {self.session_id} started audio input stream")

        except Exception as e:
            logger.error(f"Session {self.session_id} failed to initialize audio input: {str(e)}")
            await self.websocket.send_json({
                "status": "error",
                "message": f"Failed to initialize microphone: {str(e)}"
            })
            return

        logger.info(f"Session {self.session_id} listening for voice input...")
        await self.websocket.send_json({"status": "listening", "message": "Listening for voice input..."})

        # Add a timeout for silence
        silence_timeout = 10.0  # seconds
        last_speech_time = time.time()
        consecutive_silence_frames = 0
        required_silence_frames = int(1.5 * 1000 / self.frame_duration_ms)  # 1.5 seconds of silence required

        try:
            while self.is_active:
                try:
                    frame = await asyncio.wait_for(self.input_audio_queue.get(), timeout=0.5)
                    logger.debug(f"Session {self.session_id} received audio frame of size {len(frame)} bytes")
                except asyncio.TimeoutError:
                    # Check if we've been silent for too long
                    if self.speaking_detected and time.time() - last_speech_time > silence_timeout:
                        logger.info(f"Session {self.session_id} silence timeout reached")
                        if self.frames:
                            logger.info(f"Session {self.session_id} processing voice input after silence timeout...")
                            await self._process_audio(self.frames)
                        self.frames = []
                        self.speaking_detected = False
                        self.silent_frames = 0
                        consecutive_silence_frames = 0
                        last_speech_time = time.time()
                    continue
                    
                is_speech = self._voice_activity_detection(frame)
                logger.debug(f"Session {self.session_id} VAD result: {'speech' if is_speech else 'silence'}")

                if is_speech:
                    if not self.speaking_detected:
                        logger.info(f"Session {self.session_id} detected start of speech")
                        await self.websocket.send_json({"status": "speech_detected", "message": "Speech detected"})
                        
                        # Cancel the current TTS task and flush audio immediately when user starts speaking
                        if hasattr(self, "current_tts_task") and self.current_tts_task:
                            # Signal the generator loop to stop
                            if self.current_tts_cancel:
                                self.current_tts_cancel.set()

                        # Clear the output audio queue and flush player immediately
                        if self.player:
                            self.loop.call_soon_threadsafe(self.player.flush)
                            
                        # Clear any queued audio
                        while not self.output_audio_queue.empty():
                            try:
                                self.output_audio_queue.get_nowait()
                                self.output_audio_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        
                    self.speaking_detected = True
                    self.silent_frames = 0
                    consecutive_silence_frames = 0
                    self.frames.append(frame)
                    last_speech_time = time.time()
                    logger.debug(f"Session {self.session_id} added speech frame, total frames: {len(self.frames)}")
                elif self.speaking_detected:
                    self.silent_frames += 1
                    consecutive_silence_frames += 1
                    self.frames.append(frame)
                    logger.debug(f"Session {self.session_id} added silence frame, silent_frames: {self.silent_frames}/{self.frames_until_silence}")

                    if consecutive_silence_frames >= required_silence_frames:
                        # Process the voice input
                        if self.frames:
                            total_audio_duration = len(self.frames) * self.frame_duration_ms / 1000.0
                            logger.info(f"Session {self.session_id} processing voice input: {len(self.frames)} frames, duration: {total_audio_duration:.2f}s")
                            await self.websocket.send_json({"status": "processing", "message": "Processing speech..."})
                            await self._process_audio(self.frames)

                        self.frames = []
                        self.speaking_detected = False
                        self.silent_frames = 0
                        consecutive_silence_frames = 0
                        last_speech_time = time.time()
                        logger.info(f"Session {self.session_id} reset audio state after processing")
        except asyncio.CancelledError:
            logger.info(f"Listener task cancelled for session {self.session_id}")
        finally:
            if self.stream:
                logger.info(f"Session {self.session_id} stopping audio input stream")
                self.stream.stop()
                self.stream.close()
                logger.info(f"Session {self.session_id} closed audio input stream")
    
    async def _process_audio(self, frames):
        try:
            total_bytes = sum(len(f) for f in frames)
            total_audio_duration = len(frames) * self.frame_duration_ms / 1000.0
            logger.info(f"[{self.session_id}] Processing {len(frames)} frames, total size: {total_bytes} bytes, duration: {total_audio_duration:.2f}s")
            
            # Check if the audio is long enough to be meaningful speech
            if len(frames) < self.min_speech_frames:
                logger.warning(f"[{self.session_id}] Audio too short ({total_audio_duration:.2f}s), ignoring")
                await self.websocket.send_json({"status": "warning", "message": "Speech too short, ignoring"})
                return
            
            audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"[{self.session_id}] Audio converted to numpy array, shape: {audio.shape}, dtype: {audio.dtype}")

            # Check if audio has enough energy to be meaningful
            energy = np.linalg.norm(audio) / np.sqrt(audio.size)
            if energy < self.min_speech_energy:
                logger.warning(f"[{self.session_id}] Audio energy too low ({energy:.4f}), ignoring")
                await self.websocket.send_json({"status": "warning", "message": "Speech too quiet, ignoring"})
                return
            
            async with self.mlx_lock:
                try:
                    audio_mx = mx.array(audio)
                    if audio_mx is None:
                        raise ValueError("Failed to create MLX array from audio data")
                    logger.debug(f"[{self.session_id}] MLX array created, shape: {audio_mx.shape}")
                        
                    logger.info(f"[{self.session_id}] Starting transcription at {time.time()}")
                    result = await asyncio.to_thread(self.stt.generate, audio_mx)
                    if result is None or not hasattr(result, 'text'):
                        raise ValueError("Failed to generate transcription")
                        
                    text = result.text.strip()
                    logger.info(f"[{self.session_id}] Raw transcription: '{text}'")
                    
                    # Basic validation - reject empty or very short transcriptions
                    if not text or len(text.strip()) < 2:
                        logger.warning(f"[{self.session_id}] Transcription too short or empty: '{text}'")
                        await self.websocket.send_json({"status": "warning", "message": "No clear speech detected"})
                        return
                    
                    # Filter out common hallucinations that occur with noise
                    common_hallucinations = {
                        "thank you", "thanks", "you're welcome", "welcome", "bye", "goodbye", 
                        "hello", "hi", "yeah", "yes", "no", "okay", "ok", "um", "uh"
                    }
                    
                    text_lower = text.lower().strip()
                    
                    # If the transcription is a short common phrase and energy is low, it's likely a hallucination
                    if text_lower in common_hallucinations and energy < 0.03:
                        logger.warning(f"[{self.session_id}] Likely hallucination detected with low energy: '{text}' (energy: {energy:.4f})")
                        await self.websocket.send_json({"status": "warning", "message": "Filtered out possible noise hallucination"})
                        return
                    
                    # Reject if transcription is too repetitive (another sign of hallucination)
                    words = text_lower.split()
                    if len(words) > 1 and len(set(words)) == 1:  # All words are the same
                        logger.warning(f"[{self.session_id}] Repetitive transcription detected: '{text}'")
                        await self.websocket.send_json({"status": "warning", "message": "Repetitive speech ignored"})
                        return
                    
                    # Simple echo detection - only check for very similar recent responses
                    if self.conversation_history:
                        last_response = self.conversation_history[-1].get('content', '') if self.conversation_history[-1].get('role') == 'assistant' else ''
                        if last_response:
                            # Simple similarity check - if transcription is very similar to last response, it might be echo
                            response_lower = last_response.lower().strip()
                            
                            # Only reject if it's very similar (high threshold)
                            if text_lower in response_lower or response_lower in text_lower:
                                if len(text_lower) > 5 and len(text_lower) / len(response_lower) > self.echo_similarity_threshold:
                                    logger.warning(f"[{self.session_id}] Possible echo detected, ignoring: '{text}'")
                                    await self.websocket.send_json({"status": "warning", "message": "Echo detected, ignoring"})
                                    return
                    
                    logger.info(f"[{self.session_id}] Valid transcription: '{text}'")
                    await self.websocket.send_json({"status": "transcribed", "message": f"transcribed: {text}"})
                    await self.transcription_queue.put(text)
                    
                except Exception as e:
                    logger.error(f"[{self.session_id}] Transcription error: {str(e)}", exc_info=True)
                    await self.websocket.send_json({"status": "error", "message": f"Transcription failed: {str(e)}"})
                    
        except Exception as e:
            logger.error(f"[{self.session_id}] Audio processing error: {str(e)}", exc_info=True)
            await self.websocket.send_json({"status": "error", "message": f"Error processing audio: {str(e)}"})
    
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

            # Only keep the last 2 messages for context
            if len(self.conversation_history) >= 2:
                self.conversation_history = self.conversation_history[-2:]
            
            # Add the new user message
            self.conversation_history.append({"role": "user", "content": text})
            
            logger.info(f"Session {self.session_id} conversation history BEFORE LLM call: {self.conversation_history}")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. You always respond with short sentences and never use punctuation like parentheses or colons that wouldn't appear in conversational speech.",
                }
            ] + self.conversation_history
            
            logger.info(f"Session {self.session_id} sending to LLM: {messages}")

            # Generate response using Ollama
            response = await self.ollama_client.chat(
                model=self.llm_model,
                messages=messages,
                stream=False
            )
            
            response_text = response["message"]["content"].strip()
            
            # Add the assistant's response to history
            self.conversation_history.append({"role": "assistant", "content": response_text})

            logger.info(f"Session {self.session_id} LLM response: {response_text}")
            logger.info(f"Session {self.session_id} conversation history AFTER LLM call: {self.conversation_history}")

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
        try:
            logger.info(f"[{self.session_id}] Starting speech synthesis for text: '{text}'")
            await self.websocket.send_json({"status": "speaking", "message": "Speaking response..."})
            
            # Set system speaking flag to prevent audio processing during speech
            self.system_is_speaking = True
            
            # Get the speed for the selected voice
            speed = VOICE_SPEEDS.get(self.voice, 1.0)
            logger.debug(f"[{self.session_id}] Using voice speed: {speed}")
            
            loop = asyncio.get_running_loop()
            
            # Create an audio queue for streaming output
            audio_queue = asyncio.Queue()
            
            # Initialize player if not already done
            if not self.player:
                logger.info(f"[{self.session_id}] Initializing audio player")
                self.player = AudioPlayer(sample_rate=self.output_sample_rate)
            
            # Run TTS generation in a separate thread to avoid blocking
            def _tts_stream():
                try:
                    timestamp = int(time.time())
                    output_path = f"tts_{timestamp}.wav"
                    logger.info(f"[{self.session_id}] Generating TTS audio to {output_path}")
                    
                    try:
                        generate_audio(
                            text=text,
                            voice=self.voice,
                            speed=speed,
                            lang_code="a",
                            file_prefix=f"tts_{timestamp}",
                            audio_format="wav",
                            sample_rate=24000,
                            join_audio=True,
                            verbose=True,
                            max_new_tokens=128,
                            temperature=0.0,
                            top_p=0.9
                        )
                        
                        if os.path.exists(output_path):
                            logger.info(f"[{self.session_id}] TTS audio file generated successfully")
                            with open(output_path, 'rb') as f:
                                wav_data = f.read()
                            
                            audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
                            logger.debug(f"[{self.session_id}] Audio converted to numpy array, shape: {audio_np.shape}")
                            
                            loop.call_soon_threadsafe(audio_queue.put_nowait, audio_np)
                            
                            try:
                                os.remove(output_path)
                                logger.debug(f"[{self.session_id}] Temporary TTS file removed")
                            except Exception as e:
                                logger.warning(f"[{self.session_id}] Could not remove temporary file {output_path}: {e}")
                        else:
                            raise FileNotFoundError(f"Generated audio file not found: {output_path}")
                            
                    except Exception as e:
                        logger.error(f"[{self.session_id}] TTS generation error: {e}", exc_info=True)
                        raise
                        
                except Exception as e:
                    logger.error(f"[{self.session_id}] TTS stream error: {e}", exc_info=True)
                finally:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)
                    logger.info(f"[{self.session_id}] TTS stream completed")
            
            # Start TTS generation in a thread
            tts_thread = asyncio.create_task(asyncio.to_thread(_tts_stream))
            logger.info(f"[{self.session_id}] TTS generation thread started")
            
            # Process audio chunks as they become available
            while True:
                try:
                    audio_chunk = await audio_queue.get()
                    
                    if audio_chunk is None:
                        logger.info(f"[{self.session_id}] End of audio stream received")
                        break
                        
                    if not self.is_active:
                        logger.info(f"[{self.session_id}] Session no longer active, stopping audio playback")
                        break
                        
                    await self.output_audio_queue.put(audio_chunk)
                    logger.debug(f"[{self.session_id}] Audio chunk queued for playback")
                    
                    if cancel_event and cancel_event.is_set():
                        logger.info(f"[{self.session_id}] TTS cancelled by event")
                        break
                except Exception as e:
                    logger.error(f"[{self.session_id}] Error processing audio chunk: {e}", exc_info=True)
                    break
            
            await tts_thread
            logger.info(f"[{self.session_id}] TTS processing completed")
            
            # Turn off speaking flag and set a simple cooldown
            self.system_is_speaking = False
            self.speech_end_time = time.time()  # Just record when speech ended, no cooldown
            
            logger.info(f"[{self.session_id}] Speech synthesis completed, returning to listening mode")
            
            if self.is_active:
                await self.websocket.send_json({"status": "listening", "message": "Listening for voice input..."})
            
        except asyncio.CancelledError:
            logger.info(f"[{self.session_id}] Speech synthesis cancelled")
        except Exception as exc:
            logger.error(f"[{self.session_id}] Speech synthesis error: {exc}", exc_info=True)
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
                
                # Check if session is still active before playing audio
                if not self.is_active:
                    break
                    
                self.player.queue_audio(audio)
                self.output_audio_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Audio output processor task cancelled for session {self.session_id}")
        finally:
            if self.player:
                try:
                    logger.info(f"[{self.session_id}] Flushing audio player in output processor cleanup")
                    self.player.flush()
                except Exception as e:
                    logger.error(f"[{self.session_id}] Error flushing audio player in cleanup: {e}")

async def handle_sts_session(websocket: WebSocket, session_id: str = None, llm_model: str = None, voice: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # Default values if not provided
    if not llm_model:
        llm_model = "llama3"
    if not voice:
        voice = "af_heart"
        
    logger.info(f"[{session_id}] New STS session request received with model={llm_model}, voice={voice}")
    await websocket.accept()
    session = None
    
    try:
        session = STSSession(
            session_id=session_id,
            websocket=websocket,
            llm_model=llm_model,
            voice=voice
        )
        active_sessions[session_id] = session
        logger.info(f"[{session_id}] Session created and added to active_sessions")
        
        await session.start()
        logger.info(f"[{session_id}] Session started successfully")
        
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"[{session_id}] Received command: {data}")
                
                if data == "stop":
                    logger.info(f"[{session_id}] Stop command received")
                    if session:
                        await session.stop()
                        del active_sessions[session_id]
                        logger.info(f"[{session_id}] Session removed from active_sessions")
                    try:
                        if websocket.client_state.value == 1:  # WebSocket is still open
                            await websocket.send_json({"status": "stopped", "message": "STS session stopped"})
                            await websocket.close(code=1000)
                            logger.info(f"[{session_id}] WebSocket closed normally")
                    except Exception as e:
                        logger.debug(f"[{session_id}] WebSocket already closed: {e}")
                    break
                elif data.startswith("config:"):
                    config_str = data[7:]
                    try:
                        config = json.loads(config_str)
                        msg = {"status": "config", "message": "Configuration updated"}
                        if "voice" in config:
                            session.voice = config["voice"]
                            logger.info(f"[{session_id}] Voice updated to {session.voice}")
                        if "model" in config:
                            session.llm_model = config["model"]
                            logger.info(f"[{session_id}] Model updated to {session.llm_model}")
                    except Exception as e:
                        msg = {"status": "error", "message": f"Invalid configuration: {str(e)}"}
                        logger.error(f"[{session_id}] Configuration error: {e}", exc_info=True)
                
                await websocket.send_json(msg)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    logger.info(f"[{session_id}] WebSocket cancelled")
                    break
                logger.error(f"[{session_id}] WebSocket error: {str(e)}", exc_info=True)
                break
    except Exception as e:
        logger.error(f"[{session_id}] Error in STS session: {str(e)}", exc_info=True)
    finally:
        if session and session_id in active_sessions:
            logger.info(f"[{session_id}] Cleaning up session in finally block")
            await active_sessions[session_id].stop()
            del active_sessions[session_id]
            logger.info(f"[{session_id}] Session removed from active_sessions in finally block")
        
        try:
            if websocket.client_state.value == 1:  # WebSocket is still open
                await websocket.close(code=1000)
                logger.info(f"[{session_id}] WebSocket closed in finally block")
        except Exception as e:
            logger.debug(f"[{session_id}] WebSocket already closed in finally block: {e}")
