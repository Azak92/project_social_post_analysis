import unittest
from collections import UserDict
from training.smoke_2b import encode_example

class TokenizerStub:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return UserDict(input_ids=[1,2,3])
    def encode(self,text,**kwargs):return [4,5]
    def convert_tokens_to_ids(self,text):return 6

class TrainingTests(unittest.TestCase):
    def test_only_answer_and_end_token_are_supervised(self):
        ids,labels=encode_example(TokenizerStub(),'An input',50,'An answer')
        self.assertEqual(ids,[1,2,3,4,5,6])
        self.assertEqual(labels,[-100,-100,-100,4,5,6])

    def test_no_silent_truncation(self):
        class LongTokenizer(TokenizerStub):
            def encode(self,*args,**kwargs):return [4]*512
        with self.assertRaises(ValueError):encode_example(LongTokenizer(),'input',0,'answer')

if __name__=='__main__':unittest.main()
