"""Separate familiar-content fit from fresh-input transfer; no semantic claims."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work/training/controlled-wide-v3'

def assess():
    report=json.loads((RUN/'report.json').read_text())
    if not report.get('complete'): raise RuntimeError('Evaluation incomplete')
    from training.assess_pilot import assess as basic
    basic(RUN)
    result=json.loads((RUN/'diagnostics.json').read_text())
    cases=json.loads((RUN/'cases.json').read_text(encoding='utf-8'))
    grouped={}
    for mode in ('before','after'):
        grouped[mode]={}
        for category in ('training-fit','transfer-probe'):
            selected=[c for c in cases if c['category']==category]
            rows=[];identical=0;target_matches=0
            for c in selected:
                outputs=[json.loads((RUN/mode/f'{c["id"]}-{level}.json').read_text(encoding='utf-8')) for level in (0,25,50,75,100)]
                identical+=len({r['text'] for r in outputs})==1
                rows+=outputs
                if category=='training-fit':target_matches+=sum(r['text']==c['targets'][str(r['level'])] for r in outputs)
            grouped[mode][category]=dict(cases=len(selected),outputs=len(rows),identical_at_all_levels=identical,
                exact_input_copies=sum(r['text']==r['case']['text'] for r in rows),
                length_limited=sum(r['length_limited'] for r in rows),
                exact_reviewed_target_matches=target_matches if category=='training-fit' else None)
    result['by_category']=grouped
    result['limitations']+=' Training-fit outputs are memorization diagnostics. Transfer probes are small and lack independent ratings; none establish generalization alone.'
    (RUN/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(grouped,indent=2))

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    assess()
