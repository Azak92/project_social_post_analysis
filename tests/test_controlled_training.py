import unittest
from training.controlled_data import dataset, cases, IDS

class ControlledTrainingTests(unittest.TestCase):
    def test_balanced_selection_and_disjoint_probes(self):
        rows=dataset();fixed=cases()
        self.assertEqual(len(rows['train']),80)
        self.assertEqual(len(fixed),16)
        for source in IDS:
            self.assertEqual({(r['view'],r['level']) for r in rows['train'] if r['source_id']==source},
                {(v,l) for v in ('plain','styled') for l in (0,25,50,75,100)})
        train_inputs={r['messages'][1]['content'] for r in rows['train']}
        import json
        train_text={json.loads(x)['text'] for x in train_inputs}
        for c in fixed:
            if c['category']=='transfer-probe': self.assertNotIn(c['text'],train_text)
        self.assertFalse({r['group'] for r in rows['train']} & {r['group'] for r in rows['validation']})

    def test_corrected_targets_match_fit_review(self):
        targets={c['id'].removeprefix('fit_'):c['targets'] for c in cases() if c['category']=='training-fit'}
        for r in dataset()['train']:
            self.assertEqual(r['messages'][-1]['content'],targets[r['source_id']][str(r['level'])])
        self.assertIn('for repairing',targets['new_payment']['25'])
        self.assertIn('the proposal',targets['new_council']['75'])
