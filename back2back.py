from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from enum import Enum


class Language(Enum):
    PT = 1


def translation_pipeline_from_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    enpt_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
    return enpt_pipeline


class Back2BackTranslator:
    def __init__(self):
        self._translators_to = dict()
        self._translators_from = dict()

        self._translators_to[Language.PT] = \
            translation_pipeline_from_model("unicamp-dl/translation-en-pt-t5")
        self._translators_from[Language.PT] = \
            translation_pipeline_from_model("unicamp-dl/translation-pt-en-t5")

    def translate_from(self, language: Language, text_in_language):
        return self._translators_from[language](text_in_language)[0]['generated_text']

    def translate_to(self, language: Language, text_en):
        return self._translators_to[language](text_en)[0]['generated_text']
