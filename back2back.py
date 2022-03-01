from transformers import AutoTokenizer, \
    AutoModelForSeq2SeqLM, pipeline
from enum import Enum
import os
import torch


class Language(Enum):
    PT = 1


def translation_pipeline_from_model(model_name, device):
    if os.path.isdir('tokens/' + model_name):
        tokenizer = AutoTokenizer.from_pretrained('tokens/' + model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained('tokens/' + model_name)

    if os.path.isdir('models/' + model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained('models/' + model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained('models/' + model_name)

    # tokenizer = tokenizer.to(device)
    # model = model.to(device)

    enpt_pipeline = pipeline('text2text-generation',
                             model=model,
                             tokenizer=tokenizer,
                             device=0)
    return enpt_pipeline


class Back2BackTranslator:
    def __init__(self, device):
        self._device = device
        self._translators_to = dict()
        self._translators_from = dict()

        en_pt_translator_model_name = "unicamp-dl/translation-en-pt-t5"
        pt_en_translator_model_name = "unicamp-dl/translation-pt-en-t5"
        self._translators_to[Language.PT] = \
            translation_pipeline_from_model(
                en_pt_translator_model_name, device)
        self._translators_from[Language.PT] = \
            translation_pipeline_from_model(
                pt_en_translator_model_name, device)

    def translate_from(self, language: Language, text_in_language):
        return self._translators_from[language](text_in_language)[0][
            'generated_text']

    def translate_to(self, language: Language, text_en):
        return self._translators_to[language](text_en)[0]['generated_text']

    def translate_back2back(self, language: Language, text_en):
        text_in_language = self.translate_to(language, text_en)
        return self.translate_from(language, text_in_language)
