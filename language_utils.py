# language_utils.py
import requests

def fetch_available_translation_pairs():
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
        available_languages = set()

        for item in data.get("pipelineResponse", []):
            lang_info = item.get("config", {}).get("language", {})
            src = lang_info.get("sourceLanguage")
            tgt = lang_info.get("targetLanguage")
            if src and tgt:
                available_languages.add((src, tgt))

        return sorted(list(available_languages))  # e.g., [('en', 'hi'), ('en', 'ta'), ...]
    
    except Exception as e:
        print(f"Failed to fetch language pairs: {e}")
        return []
