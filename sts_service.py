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
        self.silence_threshold = config.get("silence_threshold", 0.03)
        self.silence_duration = config.get("silence_duration", 1.0)
        self.input_sample_rate = config.get("input_sample_rate", 16000)
        self.output_sample_rate = config.get("output_sample_rate", 24000)
        self.streaming_interval = config.get("streaming_interval", 3)
        self.frame_duration_ms = config.get("frame_duration_ms", 30)
        self.vad_mode = config.get("vad_mode", 3)
        
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
        self.speech_cooldown_period = 4.0  # Increased to 4 seconds cooldown after speech ends
        self.extra_caution_needed = False  # Flag for suspicious speech timing
        
        # Minimum speech requirements (to avoid processing very short utterances)
        self.min_speech_frames = int(1.0 * 1000 / self.frame_duration_ms)  # Minimum 1.0 seconds of speech required
        self.min_speech_energy = 0.06  # Increased minimum energy level for meaningful speech
        
        # Common filler words and short responses that should be filtered out
        self.filler_words = {
            "um", "uh", "hmm", "ah", "er", "like", "so", "you know", "right", 
            "okay", "ok", "yes", "no", "yeah", "nope", "sure", "thanks", 
            "thank you", "got it", "i see", "i understand", "alright", "all right"
        }
        
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
        logger.info(f"[{self.session_id}] Cancelling all tasks")
        
        # Cancel all tasks
        for task in self.tasks:
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
            
            # Stop audio player and clear queue
            if self.player:
                try:
                    logger.info(f"[{self.session_id}] Stopping audio player")
                    self.player.stop()
                    # Clear any remaining audio in the queue
                    while not self.output_audio_queue.empty():
                        try:
                            self.output_audio_queue.get_nowait()
                            self.output_audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    logger.info(f"[{self.session_id}] Audio player stopped and queue cleared")
                except Exception as e:
                    logger.error(f"[{self.session_id}] Error stopping audio player: {str(e)}")
            
        # Clear conversation history
        self.conversation_history = []
        logger.info(f"[{self.session_id}] Conversation history cleared")
            
        logger.info(f"[{self.session_id}] Session fully stopped")
        
        # Try to send final message if websocket is still open
        try:
            await self.websocket.send_json({"status": "stopped", "message": "STS session ended"})
            logger.info(f"[{self.session_id}] Stop confirmation sent to client")
        except Exception as e:
            logger.error(f"[{self.session_id}] Error sending stop confirmation: {str(e)}")
    
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
            # First check energy level with a much stricter threshold
            if isinstance(frame, bytes):
                audio_np = np.frombuffer(frame, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                audio_np = frame

            # Calculate energy of the frame
            energy = np.linalg.norm(audio_np) / np.sqrt(audio_np.size)
            
            # Check if we're in a suspicious timing window after system speech
            current_time = time.time()
            time_since_system_speech = current_time - self.speech_end_time
            is_suspicious_timing = 0 < time_since_system_speech < 1.5
            
            # Adjust threshold based on timing - if it's suspiciously close to system speech, use a stricter threshold
            energy_threshold = self.silence_threshold * (1.5 if is_suspicious_timing else 0.3)
            
            # First check: energy must be above threshold
            if energy < energy_threshold:
                logger.debug(f"Session {self.session_id} energy too low: {energy:.4f}, threshold: {energy_threshold:.4f}")
                return False
                
            # Second check: use VAD for more accurate speech detection
            is_speech = self.vad.is_speech(frame, self.input_sample_rate)
            
            # Apply more stringent checks for suspicious timing
            if is_suspicious_timing and is_speech:
                # Check for short noise bursts that might be microphone artifacts
                # Apply an extra high threshold during suspicious timing periods
                if energy < self.silence_threshold * 2.0:
                    logger.debug(f"Session {self.session_id} suspicious timing with moderate energy, treating as non-speech")
                    return False
            
            if not is_speech:
                logger.debug(f"Session {self.session_id} VAD result: non-speech, energy: {energy:.4f}")
                
            return is_speech
        except ValueError as e:
            logger.warning(f"Session {self.session_id} VAD error, falling back to energy detection: {str(e)}")
            # Fall back to energy-based detection with much stricter threshold
            return not self._is_silent(frame) and energy > self.silence_threshold * 0.8
    
    def _sd_callback(self, indata, frames, _time, status):
        if not self.is_active:
            return
            
        # Skip audio processing if system is speaking or in cooldown period
        current_time = time.time()
        if self.system_is_speaking or current_time < self.speech_end_time:
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
        # Log all available devices and the default input device
        try:
            logger.info(f"Session {self.session_id} available audio devices:")
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                logger.info(f"  Device {idx}: {dev['name']} (max input channels: {dev['max_input_channels']}, max output channels: {dev['max_output_channels']})")
            # Select a physical microphone (not virtual/loopback/system)
            physical_input_index = None
            for idx, dev in enumerate(devices):
                name = dev['name'].lower()
                if (
                    dev['max_input_channels'] > 0 and
                    not any(x in name for x in ["virtual", "loopback", "blackhole", "monitor", "output"])
                ):
                    physical_input_index = idx
                    logger.info(f"Session {self.session_id} selected physical input device: {dev['name']} (index {idx})")
                    break
            if physical_input_index is None:
                logger.warning(f"Session {self.session_id} could not find a physical input device, falling back to default.")
                physical_input_index = sd.default.device[0]
            else:
                sd.default.device = (physical_input_index, sd.default.device[1])
            input_device_info = sd.query_devices(physical_input_index)
            logger.info(f"Session {self.session_id} using input device index: {physical_input_index}, name: {input_device_info['name']}")
        except Exception as e:
            logger.warning(f"Session {self.session_id} Could not query audio devices: {e}")
            physical_input_index = None
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                blocksize=frame_size,
                channels=1,
                dtype="int16",
                callback=self._sd_callback,
                device=physical_input_index if physical_input_index is not None else None,
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

                # If system just finished speaking, check for suspicious timing
                current_time = time.time()
                time_since_system_speech = current_time - self.speech_end_time
                is_suspiciously_close_to_system_speech = 0 < time_since_system_speech < 1.0
                
                if is_speech:
                    if not self.speaking_detected:
                        # Check if this speech started suspiciously soon after system speech
                        if is_suspiciously_close_to_system_speech:
                            logger.warning(f"Session {self.session_id} speech detected too soon after system speech ({time_since_system_speech:.2f}s), being cautious")
                            # Don't immediately reject, but set a flag to be extra cautious during processing
                            self.extra_caution_needed = True
                        else:
                            logger.info(f"Session {self.session_id} detected start of speech (time since system: {time_since_system_speech:.2f}s)")
                        
                        await self.websocket.send_json({"status": "speech_detected", "message": "Speech detected"})
                        
                    self.speaking_detected = True
                    self.silent_frames = 0
                    consecutive_silence_frames = 0  # Reset consecutive silence counter when speech is detected
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
            
            # Calculate speech timing parameters - this helps detect natural speech vs echoes
            # Speech echoes tend to start immediately after system speech, while user responses have a natural pause
            speech_timing = time.time() - self.speech_end_time
            logger.info(f"[{self.session_id}] Time since last system speech: {speech_timing:.2f}s")
            
            # If speech started very soon after system finished speaking, be more suspicious
            is_suspiciously_fast_response = 0 < speech_timing < 1.5
            
            # Apply more strict thresholds if the extra caution flag is set
            if self.extra_caution_needed:
                logger.warning(f"[{self.session_id}] Using stricter thresholds due to suspicious timing")
                # Apply more strict energy threshold for suspicious speech
                if energy < self.min_speech_energy * 1.5:  # 50% higher threshold
                    logger.warning(f"[{self.session_id}] Suspicious audio with low energy ({energy:.4f}), ignoring")
                    await self.websocket.send_json({"status": "warning", "message": "Speech pattern suspicious, ignoring"})
                    # Reset the flag
                    self.extra_caution_needed = False
                    return
            
            # Reset the caution flag for next time
            self.extra_caution_needed = False
            
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
                    
                    # Normalize the text for better comparison
                    def normalize_text(t):
                        return ' '.join(t.lower().split())
                    
                    normalized_transcription = normalize_text(text)
                    
                    # Filter out common filler words
                    if normalized_transcription.lower() in self.filler_words:
                        logger.warning(f"[{self.session_id}] Filtering out filler word/phrase: '{normalized_transcription}'")
                        await self.websocket.send_json({"status": "warning", "message": "Ignoring filler word"})
                        return
                    
                    # Get last assistant message and check for similarity
                    last_llm_response = None
                    for msg in reversed(self.conversation_history):
                        if msg['role'] == 'assistant':
                            last_llm_response = msg['content']
                            break
                    
                    if last_llm_response:
                        logger.info(f"[{self.session_id}] Last LLM response: '{last_llm_response}'")
                        normalized_response = normalize_text(last_llm_response)
                        
                        # Extract words and analyze overlap
                        transcription_words = set(normalized_transcription.split())
                        response_words = set(normalized_response.split())
                        common_words = transcription_words.intersection(response_words)
                        
                        # Calculate various similarity metrics
                        word_count = len(transcription_words)
                        common_word_count = len(common_words)
                        similarity_ratio = common_word_count / word_count if word_count > 0 else 0
                        
                        # Log the similarity analysis
                        logger.info(f"[{self.session_id}] Word count: {word_count}, Common words: {common_word_count}")
                        logger.info(f"[{self.session_id}] Similarity ratio: {similarity_ratio:.2f}")
                        logger.info(f"[{self.session_id}] Common words: {common_words}")
                        
                        # More sophisticated similarity detection - use different rules based on length
                        if word_count <= 3:
                            # For very short phrases (1-3 words), use stricter rules
                            if word_count == 1:
                                # Single-word transcriptions that match any word in the response
                                if common_words or normalized_transcription in normalized_response:
                                    logger.warning(f"[{self.session_id}] Ignoring single-word transcription: '{normalized_transcription}'")
                                    await self.websocket.send_json({"status": "warning", "message": "Too short, ignoring"})
                                    return
                            elif word_count <= 3:
                                # For 2-3 word transcriptions, if similarity is high or timing is suspicious, reject
                                if similarity_ratio > 0.4 or (similarity_ratio > 0.2 and is_suspiciously_fast_response):
                                    logger.warning(f"[{self.session_id}] Ignoring short transcription with high similarity: '{normalized_transcription}'")
                                    logger.warning(f"[{self.session_id}] Similarity ratio: {similarity_ratio:.2f}, Suspicious timing: {is_suspiciously_fast_response}")
                                    await self.websocket.send_json({"status": "warning", "message": "Ignoring possible feedback"})
                                    return
                                    
                        else:
                            # For longer phrases (4+ words)
                            
                            # Check if transcription contains a significant portion of the response
                            # or if response contains the transcription
                            contains_check = normalized_transcription in normalized_response or normalized_response in normalized_transcription
                            
                            # Check for substantial word overlap
                            word_overlap_check = (
                                similarity_ratio > 0.5 or 
                                (similarity_ratio > 0.3 and common_word_count >= 3) or
                                (is_suspiciously_fast_response and similarity_ratio > 0.2 and common_word_count >= 2)
                            )
                            
                            if contains_check or word_overlap_check:
                                logger.warning(f"[{self.session_id}] Ignoring transcription that matches last LLM response")
                                logger.warning(f"[{self.session_id}] Transcription: '{normalized_transcription}'")
                                logger.warning(f"[{self.session_id}] Last response: '{normalized_response}'")
                                logger.warning(f"[{self.session_id}] Contains check: {contains_check}, Word overlap check: {word_overlap_check}")
                                await self.websocket.send_json({"status": "warning", "message": "Ignoring system feedback"})
                                return
                except Exception as e:
                    logger.error(f"[{self.session_id}] MLX processing error: {str(e)}", exc_info=True)
                    raise

            if text:
                logger.info(f"[{self.session_id}] Transcription accepted and queued for response generation")
                await self.websocket.send_json({"status": "transcribed", "message": f"Transcribed: {text}"})
                await self.transcription_queue.put(text)
            else:
                logger.warning(f"[{self.session_id}] Empty transcription received")
                await self.websocket.send_json({"status": "warning", "message": "No speech detected in audio"})
        except Exception as e:
            logger.error(f"[{self.session_id}] Error processing audio: {str(e)}", exc_info=True)
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
            
            # Turn off speaking flag and set cooldown period
            self.system_is_speaking = False
            
            # Calculate a dynamic cooldown based on response length
            # Longer responses need longer cooldown periods
            word_count = len(text.split())
            dynamic_cooldown = min(1.0 + (word_count * 0.15), 6.0)  # Up to 6 seconds for long responses
            
            # Use the longer of the default or dynamic cooldown
            effective_cooldown = max(self.speech_cooldown_period, dynamic_cooldown)
            self.speech_end_time = time.time() + effective_cooldown
            
            logger.info(f"[{self.session_id}] Speaking ended, words: {word_count}, cooldown: {effective_cooldown:.1f}s, until {self.speech_end_time}")
            
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
                        await websocket.send_json({"status": "stopped", "message": "STS session stopped"})
                        await websocket.close(code=1000)
                        logger.info(f"[{session_id}] WebSocket closed normally")
                    except Exception as e:
                        logger.error(f"[{session_id}] Error sending stop confirmation: {e}", exc_info=True)
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
            await websocket.close(code=1000)
            logger.info(f"[{session_id}] WebSocket closed in finally block")
        except Exception as e:
            logger.error(f"[{session_id}] Error closing WebSocket in finally block: {e}", exc_info=True)
