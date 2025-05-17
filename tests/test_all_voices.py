import os
from kokoro import KPipeline
import torch

# Patch torch.load to always use weights_only=False for Kokoro voices
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# List of English voices to test
voices = [
    "af_heart.pt",
    "af_bella.pt",
    "af_jessica.pt"
]

tts_pipeline = KPipeline(lang_code='a')
device = "mps" if torch.backends.mps.is_available() else "cpu"
if hasattr(tts_pipeline, 'to'):
    tts_pipeline = tts_pipeline.to(device)

test_text = "Hello, this is a test of the Kokoro voice."

for voice_file in voices:
    voice_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kokoro_voices", voice_file)
    print(f"Testing voice: {voice_file}")
    _ = tts_pipeline.load_voice(voice_path)
    generator = tts_pipeline(test_text, voice=voice_path)
    for _, _, audio in generator:
        # Save output for manual listening
        out_path = f"test_{voice_file.replace('.pt', '')}.wav"
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        import wave
        import numpy as np
        with wave.open(out_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
        print(f"Saved: {out_path}")