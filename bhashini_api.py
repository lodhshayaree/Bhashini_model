import requests
import json
import os
import base64
from dotenv import load_dotenv
from collections import defaultdict
from pydub import AudioSegment  # Ensure pydub is installed
from io import BytesIO

from language_utils import get_script_code



# Load environment variables
load_dotenv()

# Get credentials and URL from environment
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH_TOKEN = os.getenv("BHASHINI_AUTH_TOKEN")
BHASHINI_PIPELINE_URL = os.getenv("BHASHINI_PIPELINE_URL")

# Check all necessary values exist
if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH_TOKEN, BHASHINI_PIPELINE_URL]):
    raise ValueError("Missing one or more Bhashini credentials or pipeline URL in .env file.")

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "user_id": ULCA_USER_ID,
    "api-key": ULCA_API_KEY,
    "Authorization": BHASHINI_AUTH_TOKEN
}

# SCRIPT MAP for language codes to script codes
SCRIPT_MAP = defaultdict(lambda: 'Latn')  # Default to Latin script

def fetch_supported_languages():
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    headers = {
        "Content-Type": "application/json",
        "Authorization": BHASHINI_AUTH_TOKEN
    }

    try:
        response = requests.post(url, headers=headers, json={})
        response.raise_for_status()
        data = response.json()
        language_map = {}
        for pipeline in data.get('pipelineModels', []):
            for lang in pipeline.get('languages', []):
                lang_code = lang.get('sourceLanguage')
                script_code = lang.get('sourceScriptCode')
                if lang_code and script_code:
                    language_map[lang_code] = script_code
        return language_map
    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch script codes: {e}")
        return {}

# Load dynamic language-script map
SCRIPT_MAP.update(fetch_supported_languages())

def get_script_code(language_code):
    return SCRIPT_MAP[language_code]

def bhashini_pipeline_request(payload):
    response = requests.post(
        BHASHINI_PIPELINE_URL,  # must be 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
        headers=HEADERS,
        json=payload  # use json=payload instead of data=json.dumps(payload) to automatically set content-type
    )
    response.raise_for_status()
    return response.json()

def bhashini_asr(audio_base64_string, source_language):
    payload = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                "language": {
                    "sourceLanguage": source_language
                },
                "audioFormat": "flac",
                "samplingRate": 16000
            }
        }],
        "inputData": {
            "audio": [{"audioContent": audio_base64_string}]
        }
    }
    response = bhashini_pipeline_request(payload)
    for task in response.get('pipelineResponse', []):
        if task['taskType'] == 'asr':
            return task['output'][0]['source']
    raise Exception("ASR failed.")


def bhashini_nmt(text, source_lang, target_lang):
    payload = {
        "pipelineTasks": [{
            "taskType": "translation",
            "config": {
                "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                "language": {
                    "sourceLanguage": source_lang,
                    "targetLanguage": target_lang
                }
            }
        }],
        "inputData": {
            "input": [{"source": text}]
        }

    }
    response = bhashini_pipeline_request(payload)
    for task in response.get('pipelineResponse', []):
        if task['taskType'] == 'translation':
            return task['output'][0]['target']
    raise Exception("Translation failed.")


def bhashini_tts(text, target_language, gender="female", save_to_file=False):
    payload = {
        "pipelineTasks": [{
            "taskType": "tts",
            "config": {
                "serviceId": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                "language": {
                    "sourceLanguage": target_language
                },
                "gender": gender
            }
        }],
        "inputData": {
            "text": [{"source": text}]
        }
    }
    response = bhashini_pipeline_request(payload)
    for task in response.get('pipelineResponse', []):
        if task['taskType'] == 'tts':
            audio_base64 = task['output'][0].get('audioContent')
            if audio_base64 and save_to_file:
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
                    audio_segment.export("output_audio.mp3", format="mp3")
                    print("✅ Audio saved to output_audio.mp3")
                except Exception as e:
                    print(f"⚠️ Failed to save audio: {e}")
            return audio_base64
    raise Exception("TTS failed.")

def bhashini_asr_nmt(audio_base64_string, source_language_asr, target_language_nmt):
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "modelId": "648025f27cdd753e77f461a9",
                    "language": {
                        "sourceLanguage": source_language_asr,
                        "sourceScriptCode": get_script_code(source_language_asr)
                    },
                    "domain": ["general"],
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                    "modelId": "641d1cd18ecee6735a1b372a",
                    "language": {
                        "sourceLanguage": source_language_asr,
                        "sourceScriptCode": get_script_code(source_language_asr),
                        "targetLanguage": target_language_nmt,
                        "targetScriptCode": get_script_code(target_language_nmt)
                    }
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_base64_string}]
        }
    }
    response = bhashini_pipeline_request(payload)
    for task in response.get('pipelineResponse', []):
        if task['taskType'] == 'translation':
            return task['output'][0]['target']
    raise Exception("ASR-NMT failed.")

def bhashini_asr_nmt_tts_pipeline(audio_base64_string, source_language_asr, target_language_nmt_tts):
    if source_language_asr == target_language_nmt_tts:
        raise ValueError("Source and target languages must be different for translation.")

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "modelId": "648025f27cdd753e77f461a9",
                    "language": {
                        "sourceLanguage": source_language_asr,
                        "sourceScriptCode": get_script_code(source_language_asr)
                    },
                    "domain": ["general"],
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                    "modelId": "641d1cd18ecee6735a1b372a",
                    "language": {
                        "sourceLanguage": source_language_asr,
                        "sourceScriptCode": get_script_code(source_language_asr),
                        "targetLanguage": target_language_nmt_tts,
                        "targetScriptCode": get_script_code(target_language_nmt_tts)
                    }
                }
            },
            {
                "taskType": "tts",
                "config": {
                    "serviceId": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                    "modelId": "63f7384c2ff3ab138f88c64e",
                    "language": {
                        "sourceLanguage": target_language_nmt_tts,
                        "sourceScriptCode": get_script_code(target_language_nmt_tts)
                    },
                    "gender": "female"
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_base64_string}]
        }
    }

    response = bhashini_pipeline_request(payload)

    print("Full pipeline response:")
    print(response)

    if not response or 'pipelineResponse' not in response:
        raise Exception("Invalid response from Bhashini pipeline.")

    # Extract intermediate outputs for debugging
    asr_text = None
    translated_text = None
    output_audio_base64 = None

    for task in response['pipelineResponse']:
        if task['taskType'] == 'asr':
            output = task.get('output')
            if output and isinstance(output, list):
                asr_text = output[0].get('source')
        elif task['taskType'] == 'translation':
            output = task.get('output')
            if output and isinstance(output, list):
                translated_text = output[0].get('target')
        elif task['taskType'] == 'tts':
            output = task.get('output')
            if output and isinstance(output, list):
                output_audio_base64 = output[0].get('audioContent')

    print(f"ASR Text: {asr_text}")
    print(f"Translated Text: {translated_text}")

    if not asr_text:
        raise Exception("ASR failed: No transcription returned.")
    if not translated_text:
        raise Exception("Translation failed: No target text returned.")
    if not output_audio_base64:
        raise Exception("TTS failed: No audio content returned.")

    return output_audio_base64

# 2. ADD these two new functions at the bottom of your file:
def get_pipeline_config(source_lang, target_lang, task_type):
    url = f"{BHASHINI_PIPELINE_URL}/pipeline-config"
    payload = {
        "pipelineTasks": [
            {
                "taskType": task_type,
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                        "targetLanguage": target_lang
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": "64392f96daac500b55c543cd"  # working pipelineId
        }
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()


def make_translation_pipeline_request(source_lang, target_lang, input_text, service_id):
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                        "targetLanguage": target_lang
                    },
                    "serviceId": service_id  # You must pass this from outside
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": input_text
                }
            ]
        }
    }

    return bhashini_pipeline_request(payload)