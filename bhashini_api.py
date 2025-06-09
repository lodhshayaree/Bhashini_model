"""
Unified helper module for Bhashini:
    • ASR               (speech ➜ text)
    • NMT               (text ↔ text)
    • TTS               (text ➜ speech)
    • NMT + TTS         (text ➜ text ➜ speech)
    • ASR + NMT         (speech ➜ text ➜ text)
    • ASR + NMT + TTS   (speech ➜ text ➜ text ➜ speech)
"""

import os
import json
import base64
from collections import defaultdict
from io import BytesIO

import requests
from dotenv import load_dotenv
from pydub import AudioSegment   # pip install pydub
# ──────────────────────────────────────────────────────────────────────────────
#  Environment & constants
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

ULCA_USER_ID         = os.getenv("ULCA_USER_ID")
ULCA_API_KEY         = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH_TOKEN  = os.getenv("BHASHINI_AUTH_TOKEN")
BHASHINI_PIPELINE_URL = os.getenv("BHASHINI_PIPELINE_URL")      # e.g.
# https://dhruva-api.bhashini.gov.in/services/inference/pipeline

if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH_TOKEN, BHASHINI_PIPELINE_URL]):
    raise EnvironmentError(
        "❌  Missing one or more env vars: "
        "ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH_TOKEN, BHASHINI_PIPELINE_URL"
    )

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "user_id": ULCA_USER_ID,
    "api-key": ULCA_API_KEY,
    "Authorization": BHASHINI_AUTH_TOKEN,
}

# ──────────────────────────────────────────────────────────────────────────────
#  Dynamic language‑to‑script map (falls back to Latin)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_MAP = defaultdict(lambda: "Latn")


def _fetch_supported_languages() -> dict:
    """Build {lang_code: script_code} from Bhashini registry."""
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    try:
        rsp = requests.post(url, headers={"Authorization": BHASHINI_AUTH_TOKEN}, json={})
        rsp.raise_for_status()
        lang_map = {}
        for pipe in rsp.json().get("pipelineModels", []):
            for lang in pipe.get("languages", []):
                lang_code   = lang.get("sourceLanguage")
                script_code = lang.get("sourceScriptCode")
                if lang_code and script_code:
                    lang_map[lang_code] = script_code
        return lang_map
    except Exception as err:
        print(f"⚠️  Could not fetch language map: {err}")
        return {}


SCRIPT_MAP.update(_fetch_supported_languages())


def get_script_code(lang: str) -> str:
    """Return ISO‑15924 script code; defaults to Latin."""
    return SCRIPT_MAP[lang]


# ──────────────────────────────────────────────────────────────────────────────
#  Core wrapper
# ──────────────────────────────────────────────────────────────────────────────
def _pipeline_request(payload: dict) -> dict:
    rsp = requests.post(BHASHINI_PIPELINE_URL, headers=HEADERS, json=payload, timeout=60)
    rsp.raise_for_status()
    return rsp.json()


# ------------------------------------------------------------------------------
#  Low‑level single‑task helpers
# ------------------------------------------------------------------------------

def bhashini_asr(audio_b64: str, language: str) -> str:
    """Speech‑to‑Text."""
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "language": {"sourceLanguage": language},
                    "audioFormat": "flac",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {"audio": [{"audioContent": audio_b64}]},
    }
    rsp = _pipeline_request(payload)
    for task in rsp.get("pipelineResponse", []):
        if task["taskType"] == "asr":
            return task["output"][0]["source"]
    raise RuntimeError("ASR failed: no transcription returned.")


def bhashini_nmt(text: str, src_lang: str, tgt_lang: str) -> str:
    """Text‑to‑Text translation."""
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                    "language": {"sourceLanguage": src_lang, "targetLanguage": tgt_lang},
                },
            }
        ],
        "inputData": {"input": [{"source": text}]},
    }
    rsp = _pipeline_request(payload)
    for task in rsp.get("pipelineResponse", []):
        if task["taskType"] == "translation":
            return task["output"][0]["target"]
    raise RuntimeError("Translation failed.")


def bhashini_tts(text: str, language: str, gender: str = "female",
                 save_to_file: bool = False) -> str:
    """Text‑to‑Speech (same language)."""
    if gender.lower() not in {"female", "male"}:
        gender = "female"                    # safety default

    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                    "language": {"sourceLanguage": language},
                    "gender": gender,
                },
            }
        ],
        "inputData": {"text": [{"source": text}]},
    }
    rsp = _pipeline_request(payload)
    for task in rsp.get("pipelineResponse", []):
        if task["taskType"] == "tts":
            audio_b64 = task["output"][0]["audioContent"]
            if save_to_file and audio_b64:
                _save_mp3(audio_b64, "tts_output.mp3")
            return audio_b64
    raise RuntimeError("TTS failed. : no audioContent returned.")
    




def bhashini_nmt_tts(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Text → Translated Text → TTS Audio (Base64)
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": src_lang,
                        "targetLanguage": tgt_lang
                    },
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                }
            },
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": tgt_lang
                    },
                    "serviceId": "ai4bharat/indic-tts-coqui-ml--gpu",
                    "gender": "female",
                    "samplingRate": 8000
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": text
                }
            ]
        }
    }

    rsp = _pipeline_request(payload)
    for task in rsp.get("pipelineResponse", []):
        if task["taskType"] == "tts":
            return task["audio"][0]["audioContent"]
    raise RuntimeError("NMT+TTS failed: No audio content returned.")

def bhashini_asr_nmt(audio_b64: str, src_asr_lang: str, tgt_lang: str) -> str:
    """Speech → Text → Text."""
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "language": {
                        "sourceLanguage": src_asr_lang,
                        "sourceScriptCode": get_script_code(src_asr_lang),
                    },
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            },
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                    "language": {
                        "sourceLanguage": src_asr_lang,
                        "sourceScriptCode": get_script_code(src_asr_lang),
                        "targetLanguage": tgt_lang,
                        "targetScriptCode": get_script_code(tgt_lang),
                    },
                },
            },
        ],
        "inputData": {"audio": [{"audioContent": audio_b64}]},
    }

    rsp = _pipeline_request(payload)
    for task in rsp.get("pipelineResponse", []):
        if task["taskType"] == "translation":
            return task["output"][0]["target"]
    raise RuntimeError("ASR+NMT failed.")


# ──────────────────────────────────────────────────────────────────────────────
#  Composite helper: SPEECH ➜ TEXT ➜ TEXT ➜ SPEECH
# ──────────────────────────────────────────────────────────────────────────────
def bhashini_asr_nmt_tts(audio_b64: str, src_asr_lang: str,
                         tgt_lang: str, gender: str = "female") -> str:
    if src_asr_lang == tgt_lang:
        raise ValueError("Source and target languages must differ for ASR+NMT+TTS")

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": src_asr_lang,
                        "sourceScriptCode": get_script_code(src_asr_lang),
                    },
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "audioFormat": "flac",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": src_asr_lang,
                        "sourceScriptCode": get_script_code(src_asr_lang),
                        "targetLanguage": tgt_lang,
                        "targetScriptCode": get_script_code(tgt_lang),
                    },
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                }
            },
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": tgt_lang,
                        "sourceScriptCode": get_script_code(tgt_lang),
                    },
                    "serviceId": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                    "gender": gender,
                    "samplingRate": 8000
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_b64}]
        }
    }

    rsp = _pipeline_request(payload)

    # ── Extract and log intermediate outputs for debugging ──
    for task in rsp.get("pipelineResponse", []):
        task_type = task.get("taskType", "unknown")
        output = task.get("output", [])
        if not output:
            raise RuntimeError(f"{task_type.upper()} failed: no output returned")
        if task_type == "tts":
            audio_out = output[0].get("audioContent")
            if not audio_out:
                raise RuntimeError("TTS failed: audioContent missing")
            return audio_out

    raise RuntimeError("ASR+NMT+TTS failed: no audioContent returned.")

# ------------------------------------------------------------------------------
#  Utilities
# ------------------------------------------------------------------------------

def _save_mp3(b64_string: str, filename: str) -> None:
    """Helper to quickly drop MP3 files for debugging."""
    audio_bytes = base64.b64decode(b64_string)
    AudioSegment.from_file(BytesIO(audio_bytes), format="mp3").export(filename, format="mp3")
