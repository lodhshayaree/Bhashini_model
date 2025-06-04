import sounddevice as sd
import soundfile as sf
import io
import requests
import speech_recognition as sr
import base64
import numpy as np # Needed for array manipulation in soundfile

def play_audio_from_base64(audio_base64_content):
    """
    Decodes base64 audio content and plays it.
    Assumes WAV format for simplicity, adjust if Bhashini returns other formats.
    """
    if not audio_base64_content:
        print("No audio content to play.")
        return

    try:
        audio_bytes = base64.b64decode(audio_base64_content)
        # Bhashini often returns WAV, but could be other formats.
        # sf.read can often infer from the header.
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Handle stereo vs mono for sounddevice.play
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten() # Convert mono stereo to mono
        elif data.ndim == 1:
            pass # Already mono
        else:
            # If stereo, sounddevice can handle it directly
            pass

        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio from base64: {e}")
        # Optionally, print the first few bytes to debug format issues
        # print(audio_bytes[:50])


def recognize_speech_and_encode(language='en-IN', duration=5):
    """
    Captures speech from the microphone, saves it as a WAV in memory,
    and returns its base64 encoded string.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source: # Bhashini ASR often expects 16kHz
        recognizer.adjust_for_ambient_noise(source) # Adjust for noise
        print(f"Speak now for {duration} seconds...")
        try:
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            # Convert audio to WAV in memory
            wav_data = io.BytesIO(audio.get_wav_data(convert_rate=16000, convert_width=2)) # 16kHz, 16-bit
            # Encode to base64
            encoded_string = base64.b64encode(wav_data.getvalue()).decode('utf-8')
            return encoded_string
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None
        except Exception as e:
            print(f"Error capturing speech: {e}")
            return None