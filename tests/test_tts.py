import os
import time
import threading
import queue
import sounddevice as sd
import numpy as np
import torch
from kokoro import KPipeline

class AudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.queue = queue.Queue()
        self.playing = False
        self.thread = None
        self.current_audio = None
        self.next_audio = None
        self.stream = None
        self.is_playing = False
        self.last_segment = False

    def start(self):
        # Initialize the audio stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self.stream.start()
        self.playing = True

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.playing = False

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        
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
            # Only set is_playing to False if this isn't the last segment
            if not self.last_segment:
                self.is_playing = False
        else:
            # Write a full frame and keep the rest
            outdata[:, 0] = self.current_audio[:frames]
            self.current_audio = self.current_audio[frames:]

    def play_audio(self, audio_data, is_last=False):
        # Convert PyTorch tensor to numpy array if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Set last segment flag
        self.last_segment = is_last
        
        # Add to queue
        self.queue.put(audio_data)

    def wait_for_queue_empty(self, timeout=3):
        start_time = time.time()
        while not self.queue.empty() or self.is_playing:
            if time.time() - start_time > timeout:
                print("Warning: Timeout waiting for audio queue to empty")
                break
            time.sleep(0.1)
            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if elapsed > 0 and elapsed % 5 < 0.1:
                print(f"Still waiting for audio to finish... ({elapsed:.1f}s)")

def test_kokoro_tts():
    # Initialize the pipeline with American English
    pipeline = KPipeline(lang_code='a')
    
    # Test text - using a longer passage to better hear the voice characteristics
    text = """
    This is a test of the Kokoro text to speech system. 
    We are testing different voice styles to find the one that works best.
    The voice should sound natural and clear, with good intonation and pacing.
    """
    
    # Using af_heart as it's one of the best quality voices according to VOICES.md
    voice = 'af_heart'
    print(f"\nTesting voice: {voice}")
    
    try:
        # Initialize audio player
        player = AudioPlayer()
        player.start()
        
        # Generate audio with optimal settings
        # Using a more aggressive split pattern to minimize gaps
        generator = pipeline(
            text,
            voice=voice,
            speed=1.0,
            split_pattern=r'[.!?]\s+'  # Split on sentence endings
        )
        
        # Process all segments from the generator
        last_play_end = 0
        segments = list(generator)  # Convert generator to list to know total count
        total_segments = len(segments)
        
        for i, (gs, ps, audio) in enumerate(segments):
            segment_start = time.time()
            print(f"\nSegment {i}:")
            print(f"Graphemes: {gs}")
            print(f"Phonemes: {ps}")
            
            # Calculate time since last segment ended
            if last_play_end > 0:
                gap = segment_start - last_play_end
                print(f"Time since last segment: {gap:.3f} seconds")
            
            # Play the audio
            print("Playing audio...")
            play_start = time.time()
            player.play_audio(audio, is_last=(i == total_segments - 1))
            play_end = time.time()
            last_play_end = play_end
            
            # Print timing information
            play_duration = play_end - play_start
            print(f"Playback duration: {play_duration:.3f} seconds")
            print(f"Segment {i} complete!")
        
        # Wait for all audio to finish playing with a longer timeout
        print("\nWaiting for all audio to finish playing...")
        player.wait_for_queue_empty()
        
        # Stop the audio player
        player.stop()
        print(f"\nTest complete for {voice}!")
        
    except Exception as e:
        print(f"Error with {voice}: {str(e)}")
        if 'player' in locals():
            player.stop()

if __name__ == "__main__":
    # Enable GPU acceleration for M4 Max
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    test_kokoro_tts() 