import unittest
from unittest.mock import patch
from collections import Counter
from fastapi.testclient import TestClient
from lab.app import app
from lab.archive_style import select_examples,category
from lab.prepare_pairs import choose

class ArchiveTests(unittest.TestCase):
    def test_retrieval_excludes_holdout_and_political_contamination(self):
        rows=[dict(id=str(i),text=text,split=split,category=cat,group=str(i),style_signals=3)
              for i,(text,split,cat) in enumerate([
                  ('Our excellent team won!','train','praise'),
                  ('Our excellent team won the election!','train','politics'),
                  ('Our excellent team did great work!','test','praise')])]
        with patch('lab.archive_style.corpus',return_value=(rows,Counter())):
            result=select_examples('The team did excellent work on our project.')
        self.assertEqual([r['id'] for r in result],['0'])

    def test_modes_and_reference_provenance(self):
        refs=[dict(id='1',text='A tremendous day!',url='https://truthsocial.com/1',category='praise')]
        with patch('lab.app.models',return_value={'models':['test']}),patch('lab.app.select_examples',return_value=refs),patch('lab.app.ollama',return_value={'response':'A good day!'}) as gen:
            client=TestClient(app)
            result=client.post('/api/rewrite',json=dict(text='A nice day.',model='test',mode='archive'))
            self.assertEqual(result.json()['references'],refs)
            self.assertIn('A tremendous day!',gen.call_args.args[1]['prompt'])
            result=client.post('/api/rewrite',json=dict(text='A nice day.',model='test',mode='baseline'))
            self.assertEqual(result.json()['references'],[])
            self.assertNotIn('A tremendous day!',gen.call_args.args[1]['prompt'])
            self.assertEqual(client.post('/api/rewrite',json=dict(text='x',model='test',mode='bad')).status_code,422)

    def test_pair_selection_unique_groups(self):
        rows=[dict(id=str(i),split='train',category='praise',style_signals=3,group='same') for i in range(4)]
        self.assertEqual(len(choose(rows,{'train':4})),1)

if __name__=='__main__':unittest.main()
