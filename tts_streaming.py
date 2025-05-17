"""
TTS Streaming Audio Playback Module
This module provides classes for real-time TTS audio streaming with Kokoro.
It implements an efficient playback system that minimizes lag between
text generation and audio playback by streaming audio segments as they're generated.
"""

import numpy as np
import torch
import sounddevice as sd
import time
import queue
import threading
from typing import List, Optional, Tuple, Generator

class AudioPlayer:
    """
    Real-time audio player for TTS streaming.
    Plays audio chunks as they are generated, with minimal latency.
    """
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.queue = queue.Queue()
        self.current_audio = None
        self.stream = None
        self.is_playing = False
        self.last_segment = False
        self.stop_requested = False

    def start(self):
        """Initialize and start the audio stream"""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self.stream.start()
        self.is_playing = False
        self.stop_requested = False

    def stop(self):
        """Stop and close the audio stream"""
        self.stop_requested = True
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False

    def _audio_callback(self, outdata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.stop_requested:
            outdata.fill(0)
            return
            
        if self.current_audio is None or len(self.current_audio) == 0:
            if not self.queue.empty():
                self.current_audio = self.queue.get()
                self.is_playing = True
            else:
                outdata.fill(0)
                self.is_playing = False
                return

        # Calculate how much audio we can write
        remaining = len(self.current_audio)
        if remaining <= frames:
            # Write what's left and fill the rest with zeros
            outdata[:remaining, 0] = self.current_audio
            outdata[remaining:, 0] = 0
            self.current_audio = None
        else:
            # Write a full frame and keep the rest
            outdata[:, 0] = self.current_audio[:frames]
            self.current_audio = self.current_audio[frames:]

    def play_audio(self, audio_data, is_last=False):
        """
        Add audio data to the playback queue
        
        Args:
            audio_data: Audio data as PyTorch tensor or numpy array
            is_last: Flag to indicate if this is the last segment
        """
        # Convert PyTorch tensor to numpy array if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Set last segment flag
        self.last_segment = is_last
        
        # Add to queue
        self.queue.put(audio_data)

    def is_queue_empty(self):
        """Check if the queue is empty"""
        return self.queue.empty() and not self.is_playing

    def wait_until_done(self, timeout=30):
        """Wait until all audio has been played"""
        start_time = time.time()
        while not self.is_queue_empty():
            if time.time() - start_time > timeout:
                print("Warning: Timeout waiting for audio queue to empty")
                break
            time.sleep(0.1)


class StreamingTTS:
    """
    Streaming TTS manager that efficiently handles real-time audio generation and playback.
    Uses the Kokoro TTS pipeline to generate audio and streams it in real-time.
    """
    def __init__(self, pipeline, device="mps", sample_rate=24000):
        """
        Initialize the streaming TTS manager
        
        Args:
            pipeline: Kokoro TTS pipeline instance
            device: Torch device to use ("mps", "cpu", etc.)
            sample_rate: Audio sample rate
        """
        self.pipeline = pipeline
        self.device = device
        self.sample_rate = sample_rate
        self.player = AudioPlayer(sample_rate=sample_rate)
        self.player.start()
        self.is_streaming = False
        self._thread = None
        
    def stop(self):
        """Stop all audio playback and streaming"""
        self.is_streaming = False
        if self.player:
            self.player.stop()
    
    def _stream_audio(self, generator, on_segment_callback=None):
        """Internal method to stream audio segments from a generator"""
        try:
            segments = list(generator)  # Convert generator to list to know total count
            total_segments = len(segments)
            
            for i, (gs, ps, audio) in enumerate(segments):
                if not self.is_streaming:
                    break
                
                # Call the callback if provided
                if on_segment_callback:
                    on_segment_callback(i, gs, ps, audio)
                
                # Play the audio
                self.player.play_audio(audio, is_last=(i == total_segments - 1))
                
            # Wait for audio to finish playing
            if self.is_streaming:
                self.player.wait_until_done()
                
        except Exception as e:
            print(f"Error in audio streaming: {str(e)}")
        finally:
            self.is_streaming = False
    
    # def generate_and_play(self, text, voice, speed=1.0, split_pattern=r'[.!?]\s+', 
    #                      on_segment_callback=None):
    #     """
    #     Generate audio from text and play it in streaming mode
        
    #     Args:
    #         text: The text to convert to speech
    #         voice: The voice to use (name or tensor)
    #         speed: Playback speed multiplier
    #         split_pattern: Regex pattern to split text into segments
    #         on_segment_callback: Optional callback function called for each segment
    #     """
    #     # Stop any existing streaming
    #     if self.is_streaming:
    #         self.stop()
        
    #     # Start new streaming session
    #     self.is_streaming = True
        
    #     # Generate audio in a separate thread to not block
    #     self._thread = threading.Thread(target=self._stream_audio_from_text, 
    #                                   args=(text, voice, speed, split_pattern, on_segment_callback))
    #     self._thread.daemon = True
    #     self._thread.start()
        
    def _stream_audio_from_text(self, text, voice, speed, split_pattern, on_segment_callback):
        """Internal method to generate and stream audio from text"""
        try:
            # Generate audio with the pipeline
            with torch.inference_mode():
                generator = self.pipeline(
                    text,
                    voice=voice,
                    speed=speed,
                    split_pattern=split_pattern
                )
                # Stream the generated audio
                self._stream_audio(generator, on_segment_callback)
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            self.is_streaming = False
