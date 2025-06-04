from transformers import pipeline
from bhashini_api import translate_text

ner = pipeline("ner", grouped_entities=True)

def extract_entities(text, source_lang="en"):
    if source_lang != "en":
        text = translate_text(text, src_lang=source_lang, tgt_lang="en")
    return {ent["entity_group"]: ent["word"] for ent in ner(text)}
