import unittest
from pydantic import ValidationError
from lab.app import RewriteInput
from lab.evaluation_cases import cases, LEVELS

class EvaluationTests(unittest.TestCase):
    def test_five_levels_only(self):
        for level in LEVELS:
            self.assertEqual(RewriteInput(text='x',model='local',intensity=level).intensity,level)
        for level in (-1,1,24,26,51,99,101):
            with self.assertRaises(ValidationError):RewriteInput(text='x',model='local',intensity=level)

    def test_fixed_cases_have_review_criteria(self):
        rows=cases()
        self.assertEqual(len(rows),30)
        self.assertEqual(len({r['id'] for r in rows}),30)
        self.assertTrue(all(r['text'] and r['meaning_check'] for r in rows))
        self.assertIn('robustness',{r['category'] for r in rows})

if __name__=='__main__':unittest.main()
