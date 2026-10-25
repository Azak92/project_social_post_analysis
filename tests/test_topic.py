import json
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from lab.app import app
from lab.topic import generate, copied_passage

REFS=[dict(id='1',url='https://example.com/1',text='A remarkable day, an incredible day. What a result!')]
class TopicTests(unittest.TestCase):
    def test_input_and_model_validation(self):
        client=TestClient(app)
        for topic in ('', '   ', 'x'*301):
            self.assertEqual(client.post('/api/topic',json=dict(topic=topic,model='test')).status_code,422)
        with patch('lab.app.models',return_value={'models':[]}):
            self.assertEqual(client.post('/api/topic',json=dict(topic='coffee',model='missing')).status_code,503)

    def test_retry_overlong_and_reject_truncation(self):
        with patch('lab.topic.references_for_topic',return_value=REFS),patch('lab.topic.secrets.randbelow',return_value=10):
            from unittest.mock import Mock
            call=Mock(side_effect=[{'response':'x'*281},{'response':'Coffee? Make it GREAT!'}])
            result=generate('coffee','test',call)
            self.assertEqual(result['seed'],11)
            self.assertEqual(call.call_count,2)
            self.assertEqual(json.loads(call.call_args.args[1]['prompt'])['topic'],'coffee')
            call=Mock(return_value={'response':'Partial','done_reason':'length'})
            with self.assertRaises(ValueError):generate('coffee','test',call)

    def test_copy_check_and_retry(self):
        self.assertTrue(copied_passage(REFS[0]['text'],REFS))
        self.assertFalse(copied_passage('Coffee deserves better. MUCH better!',REFS))
        with patch('lab.topic.references_for_topic',return_value=REFS):
            from unittest.mock import Mock
            call=Mock(return_value={'response':REFS[0]['text']})
            with self.assertRaises(ValueError):generate('coffee','test',call)
            self.assertEqual(call.call_count,2)

    def test_endpoint_uses_archive_and_returns_label(self):
        with patch('lab.app.models',return_value={'models':['test']}),patch('lab.topic.references_for_topic',return_value=REFS),patch('lab.app.ollama',return_value={'response':'Coffee deserves better. MUCH better!'}) as call:
            response=TestClient(app).post('/api/topic',json=dict(topic='coffee',model='test'))
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['intensity'],100)
            self.assertIn('not a real Trump post',response.json()['label'])
            prompt=json.loads(call.call_args.args[1]['prompt'])
            self.assertEqual(prompt['reference_posts'],[REFS[0]['text']])
            self.assertEqual(prompt['topic'],'coffee')

    def test_political_matches_require_topic_overlap(self):
        from lab.archive_style import select_examples
        from collections import Counter
        rows=[dict(id='1',text='Iran and the war require serious thought and a careful diplomatic response.',category='politics',split='train',group='1',style_signals=1),
              dict(id='2',text='Democrats made a terrible decision about the library and the park.',category='politics',split='train',group='2',style_signals=3)]
        with patch('lab.archive_style.corpus',return_value=(rows,Counter())):
            self.assertEqual(select_examples('war in Iran'),[])
            self.assertEqual([r['id'] for r in select_examples('war in Iran',allow_political_matches=True)],['1'])
            self.assertEqual(select_examples('coffee',allow_political_matches=True),[])
