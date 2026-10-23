import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from lab.app import app, metrics, fidelity_flags
from lab.data import connect, import_twitter, normalize

class LabTests(unittest.TestCase):
    def test_metrics_empty_and_caps(self):
        self.assertEqual(metrics('')['words'],0)
        self.assertEqual(metrics('GREAT!')['uppercase_percent'],100)

    def test_import_idempotent_and_raw_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tweets.csv'
            with p.open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['ID','Time','Tweet URL','Tweet Text'])
                w.writerow(['@realDonaldTrump','2017-01-20 06:31','https://twitter.com/realDonaldTrump/status/822421390125043713','Hello ‚Äì world'])
            with connect(Path(tmp)/'test.sqlite') as db:
                import_twitter(db,p);import_twitter(db,p)
                rows=db.execute('SELECT * FROM posts').fetchall()
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['id'],'822421390125043713')
                self.assertEqual(rows[0]['timezone'],'unknown')
                self.assertEqual(rows[0]['raw_text'],'Hello ‚Äì world')
                self.assertEqual(rows[0]['text'],'Hello – world')

    def test_api_validation_and_unavailable_model(self):
        client=TestClient(app)
        self.assertEqual(client.post('/api/analyze',json={'text':''}).status_code,422)
        self.assertEqual(client.post('/api/rewrite',json={'text':'Hello','model':'test','intensity':101}).status_code,422)
        with patch('lab.app.models',return_value={'models':[]}):
            self.assertEqual(client.post('/api/rewrite',json={'text':'Hello','model':'missing'}).status_code,503)

    def test_rewrite_sends_text_as_data(self):
        with patch('lab.app.models',return_value={'models':['test']}), patch('lab.app.ollama',return_value={'response':'A great day!'}) as mock:
            result=TestClient(app).post('/api/rewrite',json={'text':'A good day.','model':'test','intensity':75})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()['label'],'Synthetic style experiment')
            self.assertIn('Preserve facts',mock.call_args.args[1]['system'])
            self.assertFalse(mock.call_args.args[1]['think'])
            self.assertEqual(mock.call_args.args[1]['options']['num_ctx'],4096)

    def test_truncation_is_not_a_success(self):
        with patch('lab.app.models',return_value={'models':['test']}), patch('lab.app.ollama',return_value={'response':'Partial sentence','done_reason':'length'}):
            result=TestClient(app).post('/api/rewrite',json={'text':'A good day.','model':'test','intensity':75})
            self.assertEqual(result.status_code,502)

    def test_whitespace_is_rejected(self):
        self.assertEqual(TestClient(app).post('/api/rewrite',json={'text':'   ','model':'test'}).status_code,422)

    def test_fidelity_flags(self):
        flags=fidelity_flags('We may deliver 12 items. No increase.', 'We will deliver 14 items.')
        self.assertEqual(len(flags),3)
        self.assertEqual(fidelity_flags('Budget $250,000.', 'Budget $250,000!'),[])
        self.assertEqual(fidelity_flags('We may deliver 12 items. No increase.', 'We MAY deliver 12 items! NO increase!'),[])

if __name__=='__main__': unittest.main()
