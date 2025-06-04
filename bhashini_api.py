
import requests
import json
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Get credentials and URL from environment variables
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH_TOKEN = os.getenv("BHASHINI_AUTH_TOKEN")
BHASHINI_PIPELINE_URL = os.getenv("BHASHINI_PIPELINE_URL")

# Debug prints to verify loaded env variables
print(f"DEBUG: Loaded ULCA_USER_ID: '{{ULCA_USER_ID}}'")
print(f"DEBUG: Loaded ULCA_API_KEY: '{{ULCA_API_KEY}}'")
print(f"DEBUG: Loaded BHASHINI_AUTH_TOKEN: '{{BHASHINI_AUTH_TOKEN}}'")
print(f"DEBUG: Loaded BHASHINI_PIPELINE_URL: '{{BHASHINI_PIPELINE_URL}}'")

if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH_TOKEN, BHASHINI_PIPELINE_URL]):
    raise ValueError("Missing one or more Bhashini credentials or pipeline URL in .env file.")

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "user_id": ULCA_USER_ID,
    "api-key": ULCA_API_KEY,
    "Authorization": BHASHINI_AUTH_TOKEN
}

# Dynamic SCRIPT_MAP
SCRIPT_MAP = defaultdict(lambda: 'Latn')  # Default to Latin

def fetch_supported_languages():
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    headers = {
        "Content-Type": "application/json",
        "Authorization": BHASHINI_AUTH_TOKEN
    }
    payload = {}

    try:
        response = requests.post(url, headers=headers, json=payload)
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
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch language scripts dynamically: {e}")
        return {}

# Update SCRIPT_MAP from Bhashini API
fetched_map = fetch_supported_languages()
if fetched_map:
    SCRIPT_MAP.update(fetched_map)
    print(f"✅ Dynamically loaded SCRIPT_MAP with {{len(SCRIPT_MAP)}} entries.")
else:
    print("⚠️ Failed to load SCRIPT_MAP from API. Falling back to default mappings.")

def get_script_code(language_code):
    return SCRIPT_MAP[language_code]

def bhashini_pipeline_request(payload):
    print(f"\\nDEBUG: Sending request to URL: {{BHASHINI_PIPELINE_URL}}")
    print(f"DEBUG: Headers being sent: {{HEADERS}}")
    print(f"DEBUG: Payload being sent (first 500 chars): {{json.dumps(payload)[:500]}}...")
    try:
        response = requests.post(BHASHINI_PIPELINE_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"\\nHTTP error occurred: {{http_err}}")
        print(f"Status Code: {{response.status_code}}")
        print(f"Response Body: {{response.text}}")
        raise Exception(f"API Error: {{response.status_code}} - {{response.text}}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"Request error: {{req_err}}")

def bhashini_asr(audio_base64_string, source_language):
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "serviceId": "ai4bharat/conformer-hi-gpu--t4",
                    "modelId": "648025f27cdd753e77f461a9",
                    "language": {
                        "sourceLanguage": source_language,
                        "sourceScriptCode": get_script_code(source_language)
                    },
                    "domain": ["general"],
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            }
        ],
        "inputData": {
            "audio": [
                {
                    "audioContent": audio_base64_string
                }
            ]
        }
    }
    response_data = bhashini_pipeline_request(payload)
    for task_output in response_data.get('pipelineResponse', []):
        if task_output.get('taskType') == 'asr' and task_output.get('output'):
            return task_output['output'][0].get('source', '')
    raise Exception("ASR failed to get transcription.")

def bhashini_nmt(text, source_lang, target_lang):
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                    "modelId": "641d1cd18ecee6735a1b372a",
                    "language": {
                        "sourceLanguage": source_lang,
                        "sourceScriptCode": get_script_code(source_lang),
                        "targetLanguage": target_lang,
                        "targetScriptCode": get_script_code(target_lang)
                    }
                }
            }
        ],
        "inputData": {
            "text": [
                {
                    "source": text
                }
            ]
        }
    }
    response_data = bhashini_pipeline_request(payload)
    for task_output in response_data.get('pipelineResponse', []):
        if task_output.get('taskType') == 'translation' and task_output.get('output'):
            return task_output['output'][0].get('target', '')
    raise Exception("Translation failed to get output.")

def bhashini_tts(text, target_language, gender="female"):
    payload = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "serviceId": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                    "modelId": "63f7384c2ff3ab138f88c64e",
                    "language": {
                        "sourceLanguage": target_language,
                        "sourceScriptCode": get_script_code(target_language)
                    },
                    "gender": gender
                }
            }
        ],
        "inputData": {
            "text": [
                {
                    "source": text
                }
            ]
        }
    }
    response_data = bhashini_pipeline_request(payload)
    for task_output in response_data.get('pipelineResponse', []):
        if task_output.get('taskType') == 'tts' and task_output.get('output'):
            return task_output['output'][0].get('audioContent', '')
    raise Exception("TTS failed to generate audio.")

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
            "audio": [
                {
                    "audioContent": audio_base64_string
                }
            ]
        }
    }
    response_data = bhashini_pipeline_request(payload)
    for task_output in response_data.get('pipelineResponse', []):
        if task_output.get('taskType') == 'translation' and task_output.get('output'):
            return task_output['output'][0].get('target', '')
    raise Exception("ASR-NMT failed to return translation.")

def bhashini_asr_nmt_tts_pipeline(audio_base64_string, source_language_asr, target_language_nmt_tts):
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
            "audio": [
                {
                    "audioContent": audio_base64_string
                }
            ]
        }
    }
    response_data = bhashini_pipeline_request(payload)
    for task_output in response_data.get('pipelineResponse', []):
        if task_output.get('taskType') == 'tts' and task_output.get('output'):
            return task_output['output'][0].get('audioContent', '')
    raise Exception("ASR-NMT-TTS failed to generate audio.")


