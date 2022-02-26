from unittest import TestCase

from back2back import Back2BackTranslator, Language


class TestBack2BackTranslator(TestCase):
    def setUp(self):
        self.b2b = Back2BackTranslator()

    def test_translate_from(self):
        pt_text = "Meu nome é Sarah e eu vivo em Londres."
        en_text = self.b2b.translate_from(Language.PT, pt_text)
        self.assertEquals(en_text, "My name is Sarah and I live in London.")

    def test_translate_to(self):
        en_text = "My name is Sarah and I live in London"
        pt_text = self.b2b.translate_to(Language.PT, en_text)
        self.assertEquals(pt_text, "Meu nome é Sarah e eu vivo em Londres.")

    def test_translate_back2back(self):
        en_text = "My name is Sarah and I live in London."
        en_text_2 = self.b2b.translate_back2back(Language.PT, en_text)
        self.assertEquals(en_text, en_text_2)
