import copy
import json
import tempfile
import unittest
from pathlib import Path
from training.five_level_data import load_reviewed, fingerprint

class FiveLevelTests(unittest.TestCase):
    def group(self):
        g=dict(id='a',group='a',split='train',origin='original',input_plain='plain',input_styled='styled',targets={str(x):str(x) for x in (0,25,50,75,100)})
        g['review']=dict(status='approved_by_agent',content_sha256=fingerprint(g))
        return g

    def load(self,groups):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'groups.json';p.write_text(json.dumps(groups),encoding='utf-8')
            return load_reviewed(p)

    def test_both_input_views_have_all_five_targets(self):
        rows=self.load([self.group()])['train']
        self.assertEqual(len(rows),10)
        self.assertEqual({(r['view'],r['level']) for r in rows},{(v,n) for v in ('plain','styled') for n in (0,25,50,75,100)})

    def test_changed_target_invalidates_review(self):
        g=self.group();g['targets']['75']='Invented fact'
        with self.assertRaisesRegex(ValueError,'Stale'):self.load([g])

    def test_group_cannot_cross_splits(self):
        a=self.group();b=copy.deepcopy(a);b.update(id='b',split='validation');b['review']['content_sha256']=fingerprint(b)
        with self.assertRaisesRegex(ValueError,'Cross-split'):self.load([a,b])

    def test_missing_level_rejected(self):
        g=self.group();del g['targets']['25']
        with self.assertRaisesRegex(ValueError,'Missing'):self.load([g])

    def test_run_selector_rejects_arbitrary_paths(self):
        from fastapi.testclient import TestClient
        from lab.app import app
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/training-evaluation?run=../../data').status_code,422)
            self.assertEqual(client.get('/five-levels').status_code,200)
