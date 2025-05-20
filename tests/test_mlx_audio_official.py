import time
import soundfile as sf
import sounddevice as sd
from mlx_audio.tts.generate import generate_audio

VOICES = [
    "af_bella", "af_heart", "af_nicole",
    "im_nicola"
]

text = "The 2024 report detailed the 150% increase. I need to file form I-9 by 5 pm on Friday. The sign read, 'Beware! 1,000 voltz!' 😊"

for voice in VOICES:
    print(f"Generating for voice: {voice}")
    start = time.time()
    generate_audio(
        text=text,
        voice=voice,
        speed=1.0,
        lang_code="a",
        file_prefix=f"kokoro_{voice}",
        audio_format="wav",
        sample_rate=24000,
        join_audio=True,
        verbose=True
    )
    end = time.time()
    filename = f"kokoro_{voice}.wav"  # <-- FIXED
    print(f"Generation time: {end - start:.2f} seconds")
    print(f"Saved: {filename}")

    # Play the generated audio
    data, sr = sf.read(filename)
    print(f"Playing {filename} ...")
    sd.play(data, sr)
    sd.wait()
    input("Press Enter to continue to the next voice...")