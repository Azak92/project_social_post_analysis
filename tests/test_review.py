import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, MagicMock
from lab import review_pairs
from lab.app import ollama, unload, ModelInput

class ReviewTests(unittest.TestCase):
    def pair(self, ident='a', split='train', group='a'):
        p=dict(source=dict(id=ident,group=group,split=split,text='Original',url='https://truthsocial.com/a'),
               neutral='Plain',medium='Partial',review_status='approved_by_agent')
        p['review']={'content_sha256':review_pairs.fingerprint(p)}
        return p

    def test_edits_invalidate_approval(self):
        p=self.pair()
        self.assertTrue(review_pairs.eligible(p))
        for key in ('neutral','medium','source'):
            edited=copy.deepcopy(p)
            if key=='source':edited[key]['text']='Changed'
            else:edited[key]='Changed'
            self.assertFalse(review_pairs.eligible(edited))
        p['review_status']='unreviewed'
        self.assertFalse(review_pairs.eligible(p))

    def test_export_omits_unreviewed_excluded_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dest=root/'data/style/pairs';dest.mkdir(parents=True)
            records=[self.pair(str(i),group=str(i)) for i in range(4)]
            records[1]['review_status']='unreviewed';records[2]['review_status']='excluded';records[3]['neutral']='Edited'
            for i,p in enumerate(records):(dest/f'{i}.json').write_text(json.dumps(p))
            with patch.object(review_pairs,'ROOT',root),patch.object(review_pairs,'PAIR_DIR',dest),redirect_stdout(io.StringIO()):review_pairs.export()
            rows=[json.loads(r) for r in (root/'data/style/reviewed/train.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3)
            self.assertEqual({r['source_id'] for r in rows},{'0'})
            self.assertEqual({r['level'] for r in rows},{0,50,100})

    def test_cross_split_groups_fail_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            for p in [self.pair('a','train','shared'),self.pair('b','test','shared')]:
                (dest/(p['source']['id']+'.json')).write_text(json.dumps(p))
            with patch.object(review_pairs,'PAIR_DIR',dest),self.assertRaisesRegex(ValueError,'crosses splits'):review_pairs.export()

    def test_keepalive_default_and_explicit_unload(self):
        response=MagicMock();response.__enter__.return_value.read.return_value=b'{}'
        with patch('lab.app.urllib.request.urlopen',return_value=response) as url:
            ollama('generate',{'model':'local'})
            self.assertEqual(json.loads(url.call_args.args[0].data)['keep_alive'],'5m')
            ollama('generate',{'model':'local','keep_alive':0})
            self.assertEqual(json.loads(url.call_args.args[0].data)['keep_alive'],0)
        with patch('lab.app.models',return_value={'models':['local']}),patch('lab.app.ollama',return_value={}) as call:
            unload(ModelInput(model='local'))
            payload=call.call_args_list[0].args[1]
            self.assertEqual(payload['keep_alive'],0)
            self.assertNotIn('prompt',payload)

if __name__=='__main__':unittest.main()
