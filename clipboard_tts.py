print("Initializing...")

# Import necessary libraries
import keyboard  # To detect key presses
import pyperclip  # To access clipboard content
import sounddevice as sd  # To play audio
from kokoro import KPipeline  # TTS pipeline for generating speech
import time  # For delays and timing

print("Loaded libraries")

# Initialize the TTS pipeline with a language setting
pipeline = KPipeline(lang_code="a")

# Variable to keep track of the current audio stream
current_audio_stream = None


def stop_audio():
    """
    Stops any currently playing audio.
    """
    global current_audio_stream
    if current_audio_stream:
        sd.stop()
        current_audio_stream = None


def on_ctrl_c():
    """
    Handles Ctrl+C:
    - Stops any currently playing audio.
    - Copies text from the clipboard.
    - Generates speech using the Kokoro TTS engine.
    - Plays the generated speech.
    """
    print("Ctrl+C pressed")
    
    time.sleep(0.1)  # Small delay to ensure clipboard updates

    global current_audio_stream
    stop_audio()  # Stop any ongoing audio

    clipboard_text = pyperclip.paste()  # Get the copied text
    print(f"Clipboard text: {clipboard_text}")

    # Generate speech using Kokoro's TTS engine
    generator = pipeline(clipboard_text, voice="am_adam", speed=1.1)

    STOP = False  # Flag to control stopping the playback

    for _, _, audio in generator:
        sd.play(audio, 24000)  # Play generated audio with a 24kHz sample rate

        # Wait for audio to finish playing, while checking for Ctrl+C to stop early
        while sd.get_stream().active:
            if keyboard.is_pressed("ctrl+c"):
                print("STOP triggered")
                STOP = True
                stop_audio()
                break

        if STOP:
            break  # Exit the loop if STOP flag is triggered


# Bind Ctrl+C to the on_ctrl_c function
keyboard.add_hotkey("ctrl+c", on_ctrl_c)

print("Press 'Ctrl + C' to copy text and play it with TTS. Press 'Esc' to exit.")

print("Ready!")

# Wait indefinitely until 'Esc' is pressed to exit
keyboard.wait("esc")
