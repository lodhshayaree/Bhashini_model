import sounddevice as sd
import soundfile as sf
import io
import speech_recognition as sr
import base64
import numpy as np

def play_audio_from_base64(audio_base64_content):
    """
    Decodes base64 audio content and plays it.
    Assumes WAV format for simplicity. Adjust if format differs.
    """
    if not audio_base64_content:
        print("No audio content to play.")
        return

    try:
        audio_bytes = base64.b64decode(audio_base64_content)
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype='float32')
            samplerate = f.samplerate

        if data.ndim > 1:
            data = data[:, 0]  # Take only one channel if stereo

        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio from base64: {e}")

def recognize_speech_and_encode(language='en', duration=5):
    """
    Captures speech from the microphone, converts to WAV base64 string.
    The 'language' is currently not used by recognizer but can be sent to API.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=16000) as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"Speak now for {duration} seconds...")
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            wav_data = io.BytesIO(audio.get_wav_data(convert_rate=16000, convert_width=2))
            encoded_string = base64.b64encode(wav_data.getvalue()).decode('utf-8')
            return encoded_string
    except sr.WaitTimeoutError:
        print("No speech detected.")
        return None
    except Exception as e:
        print(f"Error capturing speech: {e}")
        return None