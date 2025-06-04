

import streamlit as st
from bhashini_api import bhashini_tts, bhashini_nmt, bhashini_asr, bhashini_asr_nmt_tts_pipeline, bhashini_asr_nmt # Import new function
from utils import play_audio_from_base64, recognize_speech_and_encode
import base64 # Needed for potential direct base64 audio handling

st.set_page_config(page_title="Bhashini Assistant", layout="centered")
st.title("🇮🇳 Bhashini Multilingual Voice Assistant")
with open("custom_streamlit_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
option = st.selectbox("Choose a task", [
    "Text to Speech",
    "Speech to Text",
    "Text to Text Translation",
    "Speech to Text Translation", # NEW OPTION
    "Speech to Speech Translation"
])

# --- Text to Speech ---
if option == "Text to Speech":
    text = st.text_input("Enter text to speak", "नमस्ते, आप कैसे हैं?")
    # Bhashini TTS sourceLanguage is the language of the text. Use ISO 639-1 codes.
    lang = st.text_input("Language code (e.g., en, hi, ta)", "hi")
    gender = st.selectbox("Select Gender", ["male", "female"])

    if st.button("Convert and Play"):
        if text:
            try:
                # bhashini_tts now returns base64 content
                audio_base64_content = bhashini_tts(text, lang, gender=gender)
                if audio_base64_content:
                    st.success(f"Playing audio for: '{text}' in {lang} ({gender} voice).")
                    play_audio_from_base64(audio_base64_content)
                else:
                    st.error("Received no audio content from TTS.")
            except Exception as e:
                st.error(f"TTS Error: {e}")
        else:
            st.warning("Please enter some text to convert.")

# --- Speech to Text (ASR only) ---
elif option == "Speech to Text":
    lang = st.text_input("Source Speech Language Code (e.g., en, hi)", "en")
    audio_duration = st.slider("Recording duration (seconds)", min_value=1, max_value=10, value=5)

    if st.button("Record and Transcribe"):
        st.info("Listening...")
        audio_base64 = recognize_speech_and_encode(language=lang, duration=audio_duration)

        if audio_base64:
            st.success("Audio captured. Transcribing...")
            try:
                transcribed_text = bhashini_asr(audio_base64, lang)
                st.success(f"Recognized: {transcribed_text}")
            except Exception as e:
                st.error(f"ASR Error: {e}")
        else:
            st.warning("No speech detected or error during recording.")

# --- Text to Text Translation ---
elif option == "Text to Text Translation":
    src_text = st.text_input("Source Text", "Hello, how are you?")
    src_lang = st.text_input("Source Language Code (e.g., en, hi)", "en")
    tgt_lang = st.text_input("Target Language Code (e.g., hi, ta)", "hi")
    if st.button("Translate"):
        if src_text:
            try:
                translated = bhashini_nmt(src_text, src_lang, tgt_lang)
                st.success(f"Translated Text: {translated}")
            except Exception as e:
                st.error(f"Translation Error: {e}")
        else:
            st.warning("Please enter some text to translate.")

# --- NEW: Speech to Text Translation (ASR + NMT) ---
elif option == "Speech to Text Translation":
    src_lang_asr_input = st.text_input("Source Speech Language Code (e.g., en, hi)", "en")
    tgt_lang_nmt_output = st.text_input("Target Text Language Code (e.g., hi, ta)", "hi")
    audio_duration = st.slider("Recording duration (seconds)", min_value=1, max_value=10, value=5)

    if st.button("Record and Translate to Text"):
        st.info("Listening...")
        audio_base64_for_pipeline = recognize_speech_and_encode(language=src_lang_asr_input, duration=audio_duration)

        if audio_base64_for_pipeline:
            st.success("Audio captured. Processing pipeline (ASR -> NMT)...")
            try:
                translated_text_output = bhashini_asr_nmt(
                    audio_base64_for_pipeline,
                    src_lang_asr_input,
                    tgt_lang_nmt_output
                )
                st.success(f"Translated Text: {translated_text_output}")
            except Exception as e:
                st.error(f"Speech to Text Translation Error: {e}")
        else:
            st.warning("No speech detected or error during recording.")

# --- Speech to Speech Translation (ASR + NMT + TTS) ---
elif option == "Speech to Speech Translation":
    src_lang_asr_input = st.text_input("Source Speech Language Code for recording (e.g., en, hi)", "en")
    tgt_lang_nmt_tts = st.text_input("Target Language Code (e.g., hi, ta)", "hi")
    audio_duration = st.slider("Recording duration (seconds)", min_value=1, max_value=10, value=5)

    if st.button("Translate Speech"):
        st.info("Listening...")
        audio_base64_for_pipeline = recognize_speech_and_encode(language=src_lang_asr_input, duration=audio_duration)

        if audio_base64_for_pipeline:
            st.success("Audio captured. Processing pipeline (ASR -> NMT -> TTS)...")
            try:
                final_audio_base64 = bhashini_asr_nmt_tts_pipeline(
                    audio_base64_for_pipeline,
                    src_lang_asr_input, # ASR source language
                    tgt_lang_nmt_tts    # NMT target & TTS source language
                )
                if final_audio_base64:
                    st.success(f"Playing translated speech in {tgt_lang_nmt_tts}.")
                    play_audio_from_base64(final_audio_base64)
                else:
                    st.error("Received no audio content from combined pipeline.")
            except Exception as e:
                st.error(f"Speech-to-Speech Translation Error: {e}")
        else:
            st.warning("No speech detected or error during recording.")