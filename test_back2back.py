from unittest import TestCase

from back2back import Back2BackTranslator


class TestBack2BackTranslator(TestCase):
    def setUp(self):
        # Ensure any error has its full error message printed
        self.maxDiff = None
        self.b2b = Back2BackTranslator()
        self.other_langs = [
            'pt',
            'fr',
            'de',
            'sw',
            'sp',
            'th',
            'it',
            'bg',
            'ko',
             ]

    def test_translate_from(self):
        pt_text = "Meu nome é Sarah e eu vivo em Londres."
        en_text = self.b2b.translate_from('pt', pt_text)
        print(en_text)
        self.assertEquals(en_text, "My name is Sarah and I live in London.")

    def test_translate_to(self):
        en_text = "My name is Sarah and I live in London."
        pt_text = self.b2b.translate_to('pt', en_text)
        print(pt_text)
        self.assertEquals(pt_text, "Meu nome é Sarah e eu moro em Londres.")

    def test_translate_back2back(self):
        en_text = "My name is Sarah and I live in London."
        en_text_2 = self.b2b.translate_back2back('pt', en_text)
        print(en_text_2)
        self.assertEquals(en_text, en_text_2)

    def test_translate_data_back2back(self):
        en_text = "We 're living in times of absolute insanity , as I 'm pretty sure most people are aware . For a while , waking up every day to check the news seemed to carry with it the same feeling of panic and dread that action heroes probably face when they 're trying to decide whether to cut the blue or green wire on a ticking bomb -- except the bomb 's instructions long ago burned in a fire and imminent catastrophe seems the likeliest outcome . It 's hard to stay that on-edge for that long , though , so it 's natural for people to become inured to this constant chaos , to slump into a malaise of hopelessness and pessimism "
        en_text_2 = self.b2b.translate_back2back('pt', en_text)
        print(en_text_2)
        # Make sure b2b translation differs
        self.assertNotEquals(en_text, en_text_2)

    def test_translate_to_many_languages(self):
        en_text = "We 're living in times of absolute insanity , as I 'm pretty sure most people are aware . For a while , waking up every day to check the news seemed to carry with it the same feeling of panic and dread that action heroes probably face when they 're trying to decide whether to cut the blue or green wire on a ticking bomb -- except the bomb 's instructions long ago burned in a fire and imminent catastrophe seems the likeliest outcome . It 's hard to stay that on-edge for that long , though , so it 's natural for people to become inured to this constant chaos , to slump into a malaise of hopelessness and pessimism "
        en_text_from = dict()
        for lang in self.other_langs:
            print(lang)
            en_text_from[lang] = self.b2b.translate_to(lang, en_text)

    def test_translate_btb_many_languages(self):
        en_text = "We 're living in times of absolute insanity , as I 'm pretty sure most people are aware . For a while , waking up every day to check the news seemed to carry with it the same feeling of panic and dread that action heroes probably face when they 're trying to decide whether to cut the blue or green wire on a ticking bomb -- except the bomb 's instructions long ago burned in a fire and imminent catastrophe seems the likeliest outcome . It 's hard to stay that on-edge for that long , though , so it 's natural for people to become inured to this constant chaos , to slump into a malaise of hopelessness and pessimism "
        en_text_from = dict()
        for lang in self.other_langs:
            en_text_from[lang] = self.b2b.translate_back2back(lang, en_text)

        for lang, text in en_text_from.items():
            # Make sure b2b translation differs
            self.assertNotEquals(en_text, text)
            other_texts = en_text_from.copy()
            other_texts.pop(lang)
            for lang2, text2 in other_texts.items():
                print(f"{lang}, {lang2}")
                print(text)
                print(text2)
                # Make sure b2b translation differs
                self.assertNotEquals(text, text2)
        
    def test_translate_and_sanitise(self):
        en_text = """ As Briggs &amp; Stratton celebrates its 110th anniversary year , what better way to reaffirm our commitment to the Milwaukee community than by providing $1 million to kids in need of pediatric care , "" said Rick Carpenter , vice president corporate marketing . "" We are lucky to have one of the nation 's top pediatric hospitals right here within our community and Briggs &amp; Stratton firmly stands behind its commitment to extend its support into the future . """
        en_text_2 = self.b2b.translate_back2back('pt', en_text)
        print(en_text_2)
        # Make sure b2b translation differs
        self.assertGreater(len(en_text_2), 20)
