from unittest import TestCase

from back2back import Back2BackTranslator


class TestBack2BackTranslator(TestCase):
    def setUp(self):
        # Ensure any error has its full error message printed
        self.maxDiff = None
        self.b2b = Back2BackTranslator()

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
