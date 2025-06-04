import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials and URL from environment variables
ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH_TOKEN = os.getenv("BHASHINI_AUTH_TOKEN")
BHASHINI_PIPELINE_URL = os.getenv("BHASHINI_PIPELINE_URL")

# --- DEBUGGING: Print loaded environment variables ---
print(f"DEBUG: Loaded ULCA_USER_ID: '{ULCA_USER_ID}'")
print(f"DEBUG: Loaded ULCA_API_KEY: '{ULCA_API_KEY}'")
print(f"DEBUG: Loaded BHASHINI_AUTH_TOKEN: '{BHASHINI_AUTH_TOKEN}'")
print(f"DEBUG: Loaded BHASHINI_PIPELINE_URL: '{BHASHINI_PIPELINE_URL}'")
# --- END DEBUGGING ---

# Ensure all required environment variables are loaded
if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH_TOKEN, BHASHINI_PIPELINE_URL]):
    raise ValueError("Missing one or more Bhashini credentials or pipeline URL in .env file. "
                     "Please check your .env file and ensure all values are present.")

# Define Headers for the Bhashini Pipeline API
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "user_id": ULCA_USER_ID,
    "api-key": ULCA_API_KEY,
    "Authorization": BHASHINI_AUTH_TOKEN
}

def bhashini_pipeline_request(payload):
    """
    Sends a request to the Bhashini pipeline inference API.

    Args:
        payload (dict): The JSON payload containing the pipeline request.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    # --- DEBUGGING: Print headers and payload before sending ---
    print(f"\nDEBUG: Sending request to URL: {BHASHINI_PIPELINE_URL}")
    print(f"DEBUG: Headers being sent: {HEADERS}")
    print(f"DEBUG: Payload being sent (first 500 chars): {json.dumps(payload)[:500]}...")
    # --- END DEBUGGING ---

    try:
        response = requests.post(BHASHINI_PIPELINE_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"\nHTTP error occurred: {http_err}")
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}") # THIS IS KEY FOR 500 ERRORS
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        raise Exception(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        raise Exception(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"An unexpected error occurred: {req_err}")

# --- Bhashini Task Specific Functions ---

def bhashini_asr(audio_base64_string, source_language):
    """
    Performs Automatic Speech Recognition using Bhashini pipeline.
    Returns the transcribed text.
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": source_language
                    },
                    "audioFormat": "wav",
                    "samplingRate": 16000,
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
    if response_data and response_data.get('pipelineResponse'):
        for task_output in response_data['pipelineResponse']:
            if task_output.get('taskType') == 'asr' and task_output.get('output'):
                return task_output['output'][0].get('source', '')
    raise Exception("ASR failed to get transcription or response structure unexpected.")


def bhashini_nmt(text, source_lang, target_lang):
    """
    Performs Neural Machine Translation using Bhashini pipeline.
    Returns the translated text.
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation", # <--- Corrected
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                        "targetLanguage": target_lang
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
    if response_data and response_data.get('pipelineResponse'):
        for task_output in response_data['pipelineResponse']:
            if task_output.get('taskType') == 'translation' and task_output.get('output'): # <--- Corrected
                return task_output['output'][0].get('target', '')
    raise Exception("NMT failed to get translation or response structure unexpected.")

def bhashini_tts(text, target_language, gender="male", voice_code=None):
    """
    Performs Text-to-Speech using Bhashini pipeline.
    Returns base64 encoded audio content.
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": target_language
                    },
                    "gender": gender,
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
    if response_data and response_data.get('pipelineResponse'):
        for task_output in response_data['pipelineResponse']:
            if task_output.get('taskType') == 'tts' and task_output.get('output'):
                return task_output['output'][0].get('audioContent', '')
    raise Exception("TTS failed to get audio content or response structure unexpected.")

# NEW FUNCTION FOR SPEECH TO TEXT TRANSLATION
def bhashini_asr_nmt(audio_base64_string, source_language_asr, target_language_nmt):
    """
    Performs a combined ASR -> NMT pipeline.
    Returns the translated text.
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": source_language_asr
                    },
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language_asr, # ASR output is NMT input source
                        "targetLanguage": target_language_nmt
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
    if response_data and response_data.get('pipelineResponse'):
        # Iterate through pipelineResponse to find the 'translation' task's output
        for task_output in response_data['pipelineResponse']:
            if task_output.get('taskType') == 'translation' and task_output.get('output'):
                return task_output['output'][0].get('target', '')
    raise Exception("ASR-NMT pipeline failed to get translated text or response structure unexpected.")


def bhashini_asr_nmt_tts_pipeline(audio_base64_string, source_language_asr, target_language_nmt_tts):
    """
    Performs a combined ASR -> NMT -> TTS pipeline.
    Returns the base64 encoded final audio.
    """
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": source_language_asr
                    },
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation", # <--- Corrected
                "config": {
                    "language": {
                        "sourceLanguage": source_language_asr,
                        "targetLanguage": target_language_nmt_tts
                    }
                }
            },
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": target_language_nmt_tts
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
    if response_data and response_data.get('pipelineResponse'):
        for task_output in response_data['pipelineResponse']:
            if task_output.get('taskType') == 'tts' and task_output.get('output'):
                return task_output['output'][0].get('audioContent', '')
    raise Exception("Combined ASR-NMT-TTS pipeline failed to get audio content or response structure unexpected.")