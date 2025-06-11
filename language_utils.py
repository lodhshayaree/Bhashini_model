import requests
from functools import lru_cache

# Static mappings for language codes and names
LANG_CODE_TO_NAME = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

NAME_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_NAME.items()}

@lru_cache(maxsize=1)
def fetch_available_translation_pairs():
    """
    Fetches available source-target language pairs for translation.
    Caches results for performance.
    """
    url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": "",
                        "targetLanguage": ""
                    }
                }
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        available_languages = {
            (item["config"]["language"]["sourceLanguage"],
             item["config"]["language"]["targetLanguage"])
            for item in data.get("pipelineResponse", [])
            if item.get("config", {}).get("language", {}).get("sourceLanguage") and
               item.get("config", {}).get("language", {}).get("targetLanguage")
        }
        return sorted(list(available_languages))
    except Exception as e:
        print(f"Failed to fetch language pairs: {e}")
        return []

 # Default to Latin

SCRIPT_CODES = {
    "as": "Beng", "bn": "Beng", "brx": "Deva", "doi": "Deva",
    "en": "Latn", "gu": "Gujr", "hi": "Deva", "kn": "Knda",
    "ks": "Arab", "kok": "Deva", "mai": "Deva", "ml": "Mlym",
    "mni": "Beng", "mr": "Deva", "ne": "Deva", "or": "Orya",
    "pa": "Guru", "sa": "Deva", "sat": "Olck", "sd": "Arab",
    "ta": "Taml", "te": "Telu", "ur": "Arab"
}

def get_script_code(lang_code):
    return SCRIPT_CODES.get(lang_code, "Deva")  # default to Devanagari