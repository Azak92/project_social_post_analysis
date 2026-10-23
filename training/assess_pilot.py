"""Mechanical pilot diagnostics. These counts do not replace semantic review."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work/training/archive-pilot-v1'

def assess(run=None):
    global RUN
    if run is not None: RUN=Path(run)
    report=json.loads((RUN/'report.json').read_text())
    if not report.get('complete'):raise RuntimeError('Wait for both evaluation passes to complete.')
    cases=json.loads((RUN/'cases.json').read_text(encoding='utf-8'))
    result={}
    for mode in ('before','after'):
        output=[];insensitive=[];variants=[]
        for c in cases:
            rows=[json.loads((RUN/mode/f'{c["id"]}-{level}.json').read_text(encoding='utf-8')) for level in (0,25,50,75,100)]
            distinct=len({r['text'].strip() for r in rows})
            variants.append(distinct)
            if distinct==1:insensitive.append(c['id'])
            output.extend(rows)
        result[mode]=dict(outputs=len(output),identical_at_all_five_levels=len(insensitive),insensitive_cases=insensitive,
            average_distinct_outputs_per_case=sum(variants)/len(variants),
            exact_input_copies=sum(r['text'].strip()==r['case']['text'].strip() for r in output),
            length_limited=sum(r['length_limited'] for r in output))
    result['validation_loss_relative_reduction']=(report['validation_loss_before']-report['validation_loss_after'])/report['validation_loss_before']
    result['limitations']='Exact textual differences measure responsiveness, not stylistic quality. Identical output can be appropriate for some inputs. One deterministic generation per condition.'
    (RUN/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':assess()
