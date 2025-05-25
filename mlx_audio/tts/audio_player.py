from collections import deque
from threading import Event, Lock

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self, sample_rate=24_000, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.playing = False
        self.drain_event = Event()
        self.stream = None

    def callback(self, outdata, frames, time, status):
        # Initialize with silence to prevent clicking/popping
        outdata.fill(0.0)
        filled = 0

        with self.buffer_lock:
            while filled < frames and self.audio_buffer:
                buf = self.audio_buffer[0]
                to_copy = min(frames - filled, len(buf))
                outdata[filled : filled + to_copy, 0] = buf[:to_copy]
                filled += to_copy

                if to_copy == len(buf):
                    self.audio_buffer.popleft()
                else:
                    self.audio_buffer[0] = buf[to_copy:]

            # Ensure remaining frames are filled with silence
            if filled < frames:
                outdata[filled:, 0] = 0.0
                
            if not self.audio_buffer and filled < frames:
                self.drain_event.set()

    def play(self):
        if not self.playing:
            # Pre-fill buffer with a small amount of silence to prevent clicking
            silence_frames = int(self.sample_rate * 0.01)  # 10ms of silence
            silence_buffer = np.zeros(silence_frames, dtype=np.float32)
            with self.buffer_lock:
                self.audio_buffer.append(silence_buffer)
            
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                blocksize=self.buffer_size,
                dtype='float32',  # Use float32 for better precision
            )
            self.stream.start()
            self.playing = True
            self.drain_event.clear()

    def queue_audio(self, samples):
        self.drain_event.clear()

        # Ensure samples are float32 for consistency
        if isinstance(samples, np.ndarray):
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
        else:
            samples = np.array(samples, dtype=np.float32)

        with self.buffer_lock:
            self.audio_buffer.append(samples)
        if not self.playing:
            self.play()

    def wait_for_drain(self):
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            # Add a small fade-out to prevent clicking
            with self.buffer_lock:
                if self.audio_buffer:
                    # Add a small fade-out buffer to prevent abrupt cutoff
                    fade_length = min(int(self.sample_rate * 0.005), 120)  # 5ms or 120 samples
                    fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
                    self.audio_buffer.append(fade_out * 0.0)  # Silent fade
            
            self.wait_for_drain()
            sd.sleep(50)  # Reduced sleep time

            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            
            self.playing = False
            self.stream = None

    def flush(self):
        """Discard everything and stop playback immediately."""
        if not self.playing:
            return

        with self.buffer_lock:
            self.audio_buffer.clear()

        # Use abort for immediate stopping without waiting for drain
        try:
            self.stream.abort()
            self.stream.close()
        except AttributeError:  # older sounddevice
            try:
                self.stream.stop(ignore_errors=True)
                self.stream.close()
            except:
                pass
        except Exception:
            # If abort fails, try graceful stop
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass

        self.playing = False
        self.drain_event.set()
        self.stream = None
