from enum import Enum
import requests

def translate(text, lang_src, lang_tgt):
    url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={lang_src}&tl={lang_tgt}&q={text}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
    }

    translated_text = ''
    try:
        request_result = requests.get(url, headers=headers).json()
        translated_text = request_result[0]
    except:
        raise Exception("Translation unsuccessful!")
    
    return translated_text

class Back2BackTranslator:
    def translate_from(self, lang_src, text_in_language):
        return translate(lang_src=lang_src, lang_tgt='en', text=text_in_language)

    def translate_to(self, lang_tgt, text_en):
        return translate(lang_src='en', lang_tgt=lang_tgt, text=text_en)

    def translate_back2back(self, language, text_en):
        text_in_language = self.translate_to(language, text_en)
        return self.translate_from(language, text_in_language)
