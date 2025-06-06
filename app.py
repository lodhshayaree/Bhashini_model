import streamlit as st
from bhashini_api import (
    bhashini_tts, 
    bhashini_nmt, 
    bhashini_asr, 
    bhashini_asr_nmt_tts_pipeline, 
    bhashini_asr_nmt
)
from utils import play_audio_from_base64, recognize_speech_and_encode
from language_utils import LANG_CODE_TO_NAME, NAME_TO_LANG_CODE

# App Configuration
st.set_page_config(page_title="Bhashini Assistant", layout="centered")
st.title("🇮🇳 Bhashini Multilingual Voice Assistant")

# Custom CSS
with open("custom_streamlit_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Task Selector ---
option = st.selectbox("Choose a Task", [
    "Text to Speech",
    "Speech to Text",
    "Text to Text Translation",
    "Speech to Text Translation",
    "Speech to Speech Translation"
])

# --- Text to Speech ---
if option == "Text to Speech":
    st.subheader("🗣️ Text to Speech")
    text = st.text_input("Enter Text to Convert", "नमस्ते, आप कैसे हैं?")
    lang = st.text_input("Language Code (e.g., en, hi, ta)", "hi")
    gender = st.selectbox("Select Voice Gender", ["male", "female"])

    if st.button("🔊 Convert and Play"):
        if text:
            try:
                audio_base64 = bhashini_tts(text, lang, gender=gender)
                if audio_base64:
                    st.success(f"Playing voice in {lang.upper()} ({gender}) for: \"{text}\"")
                    play_audio_from_base64(audio_base64)
                else:
                    st.error("❌ No audio returned from TTS API.")
            except Exception as e:
                st.error(f"TTS Error: {e}")
        else:
            st.warning("⚠️ Please enter some text.")

# --- Speech to Text (ASR) ---
elif option == "Speech to Text":
    st.subheader("🎙️ Speech to Text")
    lang = st.text_input("Speech Language Code (e.g., en, hi)", "en")
    duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

    if st.button("🎤 Record and Transcribe"):
        st.info("Listening...")
        audio_base64 = recognize_speech_and_encode(lang, duration)

        if audio_base64:
            st.success("Audio captured. Transcribing...")
            try:
                result = bhashini_asr(audio_base64, source_language=lang)
                st.success(f"📝 Transcribed Text: {result}")
            except Exception as e:
                st.error(f"ASR Error: {e}")
        else:
            st.warning("⚠️ No speech detected or recording failed.")

# --- Text to Text Translation ---
elif option == "Text to Text Translation":
    st.subheader("🔤 Text to Text Translation")
    src_text = st.text_input("Source Text", "Hello, how are you?")
    src_lang = st.text_input("Source Language Code (e.g., en, hi)", "en")
    tgt_lang = st.text_input("Target Language Code (e.g., hi, ta)", "hi")

    if st.button("🌐 Translate Text"):
        if src_text:
            try:
                translated = bhashini_nmt(src_text, src_lang, tgt_lang)
                st.success(f"🌍 Translated Text: {translated}")
            except Exception as e:
                st.error(f"Translation Error: {e}")
        else:
            st.warning("⚠️ Enter text to translate.")

# --- Speech to Text Translation (ASR + NMT) ---
elif option == "Speech to Text Translation":
    st.subheader("🗣️➡️📝 Speech to Text Translation")
    src_lang = st.text_input("Speech Language Code", "en")
    tgt_lang = st.text_input("Target Language Code", "hi")
    duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

    if st.button("🎤 Record and Translate"):
        st.info("Listening...")
        audio_base64 = recognize_speech_and_encode(src_lang, duration)

        if audio_base64:
            st.success("Audio captured. Processing ASR ➝ NMT...")
            try:
                translated_text = bhashini_asr_nmt(audio_base64, source_language_asr=src_lang, target_language_nmt=tgt_lang)

                st.success(f"🌍 Translated Text: {translated_text}")
            except Exception as e:
                st.error(f"Speech ➝ Text Translation Error: {e}")
        else:
            st.warning("⚠️ No audio recorded or error during recognition.")

# --- Speech to Speech Translation (ASR + NMT + TTS) ---
elif option == "Speech to Speech Translation":
    st.subheader("🗣️➡️🗣️ Speech to Speech Translation")

    src_lang = "hi"  # ASR supports only Hindi currently

    from language_utils import LANG_CODE_TO_NAME
    target_language_options = [code for code in LANG_CODE_TO_NAME.keys() if code != "hi"]
    tgt_lang = st.selectbox("Target Language Code (TTS output)", target_language_options, index=target_language_options.index("en"))

    duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

    if st.button("🎤 Translate and Speak"):
        st.info("🎙 Listening in Hindi...")
        audio_base64 = recognize_speech_and_encode(language='hi', duration=duration)

        if audio_base64:
            st.success("🎧 Audio captured. Running ASR ➝ NMT ➝ TTS pipeline...")
            try:
                output_audio_base64 = bhashini_asr_nmt_tts_pipeline(
                    audio_base64_string=audio_base64,
                    source_language_asr=src_lang,
                    target_language_nmt_tts=tgt_lang
                )

                if output_audio_base64:
                    st.success(f"🔊 Playing translated speech in {tgt_lang.upper()} ({LANG_CODE_TO_NAME[tgt_lang]})")
                    play_audio_from_base64(output_audio_base64)
                else:
                    st.error("❌ No audio received from pipeline.")
            except Exception as e:
                st.error(f"Speech ➝ Speech Translation Error: {e}")
        else:
            st.warning("⚠️ Recording failed or no speech detected.")
