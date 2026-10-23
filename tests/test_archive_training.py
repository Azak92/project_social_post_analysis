import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training import archive_pilot as pilot

class ArchiveTrainingTests(unittest.TestCase):
    def dataset(self,root):
        (root/'data/style/reviewed').mkdir(parents=True)
        (root/'data/style/pairs').mkdir()
        for split in ('train','validation','test'):
            p=dict(source=dict(text='Source text',id=split,group=split,split=split),neutral='Neutral text',medium='Medium text',review_status='approved_by_agent')
            p['review']={'content_sha256':pilot.fingerprint(p)}
            (root/f'data/style/pairs/{split}.json').write_text(json.dumps(p))
            row=dict(source_id=split,group=split,review_sha256=pilot.fingerprint(p),level=50,
                messages=[dict(role='system',content=pilot.SYSTEM),dict(role='user',content=json.dumps(dict(text=p['neutral'],intensity=50))),dict(role='assistant',content=p['medium'])])
            (root/f'data/style/reviewed/{split}.jsonl').write_text(json.dumps(row)+'\n')

    def test_rejects_changed_export_even_with_valid_source_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.dataset(root)
            with patch.object(pilot,'ROOT',root):self.assertEqual(len(pilot.load_reviewed()['train']),1)
            path=root/'data/style/reviewed/train.jsonl';row=json.loads(path.read_text());row['messages'][-1]['content']='Unreviewed replacement'
            path.write_text(json.dumps(row))
            with patch.object(pilot,'ROOT',root),self.assertRaisesRegex(ValueError,'reviewed targets'):pilot.load_reviewed()

    def test_rejects_changed_source_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.dataset(root)
            path=root/'data/style/pairs/train.json';p=json.loads(path.read_text());p['medium']='Edited after approval';path.write_text(json.dumps(p))
            with patch.object(pilot,'ROOT',root),self.assertRaisesRegex(ValueError,'Stale'):pilot.load_reviewed()

if __name__=='__main__':unittest.main()
