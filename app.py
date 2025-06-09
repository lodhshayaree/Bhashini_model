import streamlit as st

from bhashini_api import (
    bhashini_tts,
    bhashini_nmt_tts,
    bhashini_asr,
    bhashini_nmt,
    bhashini_asr_nmt,
    bhashini_asr_nmt_tts,
)
from utils import recognize_speech_and_encode, play_audio_from_base64
from language_utils import LANG_CODE_TO_NAME

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bhashini Assistant", layout="centered")
st.title("🇮🇳  Bhashini Multilingual Voice Assistant")

with open("custom_streamlit_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

TASK = st.selectbox(
    "Choose a Task",
    [
        "Text to Speech",                # TTS
        "Text to Speech Translation",    # NMT + TTS
        "Speech to Text",                # ASR
        "Text to Text Translation",      # NMT
        "Speech to Text Translation",    # ASR + NMT
        "Speech to Speech Translation",  # ASR + NMT + TTS
    ],
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) TEXT ➜ SPEECH  (same language)
# ──────────────────────────────────────────────────────────────────────────────
if TASK == "Text to Speech":
    st.subheader("🗣️  Text → Speech (same language)")

    txt   = st.text_input("Text", "नमस्ते, आप कैसे हैं?")
    lang  = st.text_input("Language code (e.g. hi, en, ta)", "hi")
    voice = st.radio("Voice gender", ["female"], horizontal=True)

    if st.button("🔊 Speak"):
        try:
            audio_b64 = bhashini_tts(txt, lang, gender=voice)
            audio_bytes = base64.b64decode(audio_b64)
            st.audio(audio_bytes, format="audio/mp3")   # browser‑native playback
            play_audio_from_base64(audio_b64)  # ✅ Hear from system speaker also

        except Exception as err:
            st.error(f"TTS failed: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) TEXT ➜ SPEECH (translate)
# ──────────────────────────────────────────────────────────────────────────────
elif TASK == "Text to Speech Translation":
    st.subheader("🔤🗣️  Text → Speech (translate)")

    src_txt = st.text_input("Source text", "Hello, how are you?")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.text_input("Source lang", "en")
    with col2:
        tgt_lang = st.text_input("Target lang", "hi")

    if st.button("🌐 Translate & Speak"):
        try:
            audio_b64 = bhashini_nmt_tts(src_txt, src_lang, tgt_lang)
            play_audio_from_base64(audio_b64)
        except Exception as err:
            st.error(f"NMT+TTS failed: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) SPEECH ➜ TEXT
# ──────────────────────────────────────────────────────────────────────────────
elif TASK == "Speech to Text":
    st.subheader("🎙️  Speech → Text")
    lang     = st.text_input("Speech language code", "en")
    duration = st.slider("Record seconds", 1, 10, 5)

    if st.button("🎤 Record"):
        audio_b64 = recognize_speech_and_encode(lang, duration)
        if audio_b64:
            try:
                text = bhashini_asr(audio_b64, lang)
                st.success(text)
            except Exception as err:
                st.error(f"ASR failed: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) TEXT ➜ TEXT
# ──────────────────────────────────────────────────────────────────────────────
elif TASK == "Text to Text Translation":
    st.subheader("🔤  Text → Text")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.text_input("Source lang", "en")
    with col2:
        tgt_lang = st.text_input("Target lang", "hi")
    src_txt = st.text_area("Source text", "Hello, how are you?")

    if st.button("🌐 Translate"):
        try:
            st.success(bhashini_nmt(src_txt, src_lang, tgt_lang))
        except Exception as err:
            st.error(f"NMT failed: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) SPEECH ➜ TEXT (translate)
# ──────────────────────────────────────────────────────────────────────────────
elif TASK == "Speech to Text Translation":
    st.subheader("🎙️📝  Speech → Text (translate)")

    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.text_input("Speech lang", "en")
    with col2:
        tgt_lang = st.text_input("Target text lang", "hi")
    duration = st.slider("Record seconds", 1, 10, 5)

    if st.button("🎤 Record & Translate"):
        audio_b64 = recognize_speech_and_encode(src_lang, duration)
        if audio_b64:
            try:
                trans_txt = bhashini_asr_nmt(audio_b64, src_lang, tgt_lang)
                st.success(trans_txt)
            except Exception as err:
                st.error(f"ASR+NMT failed: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# 6) SPEECH ➜ SPEECH
# ──────────────────────────────────────────────────────────────────────────────
elif TASK == "Speech to Speech Translation":
    st.subheader("🗣️🗣️  Speech → Speech")

    src_lang = "hi"   # ASR model is Hindi‑only on Dhruva
    tgt_lang = st.selectbox(
        "Target language",
        [code for code in LANG_CODE_TO_NAME if code != "hi"],
        index=list(LANG_CODE_TO_NAME).index("en"),
    )
    duration = st.slider("Record seconds", 1, 10, 5)

    if st.button("🎤 Record & Speak"):
        audio_b64 = recognize_speech_and_encode(src_lang, duration)
        if audio_b64:
            try:
                tts_b64 = bhashini_asr_nmt_tts(audio_b64, src_lang, tgt_lang)
                play_audio_from_base64(tts_b64)
            except Exception as err:
                st.error(f"ASR+NMT+TTS failed: {err}")
