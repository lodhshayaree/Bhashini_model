import sounddevice as sd
import soundfile as sf
import io
import speech_recognition as sr
import base64
import numpy as np

# utils.py
from pydub import AudioSegment        # new dependency
          
import tempfile

def play_audio_from_base64(audio_b64: str) -> None:
    """Play base‑64 MP3 **or** WAV through speakers."""
    if not audio_b64:
        print("No audio content to play.")
        return

    try:
        audio_bytes = base64.b64decode(audio_b64)

        # --- Detect format quickly ---
        is_mp3 = audio_bytes[:3] == b"ID3" or audio_bytes[0] & 0xFF == 0xFF

        if is_mp3:                       # MP3 → raw samples → simpleaudio
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            play_obj = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate,
            )
            play_obj.wait_done()
        else:                            # Assume WAV/PCM – delegate to soundfile
            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                data = f.read(dtype="float32")
                sd.play(data, f.samplerate)
                sd.wait()

    except Exception as e:
        print(f"Error playing audio: {e}")


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



