"""Apply explicit agent review decisions and export only unchanged reviewed pairs."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from .data import ROOT
from .app import fidelity_flags

PAIR_DIR=ROOT/'data/style/pairs'

def fingerprint(p):
    payload={k:p[k] for k in ('source','neutral','medium')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def eligible(p):
    return p.get('review_status')=='approved_by_agent' and p.get('review',{}).get('content_sha256')==fingerprint(p)

def apply(decisions_path):
    decisions=json.loads(decisions_path.read_text(encoding='utf-8'))
    history=PAIR_DIR/'history';history.mkdir(exist_ok=True)
    for ident,d in decisions.items():
        path=PAIR_DIR/(ident+'.json');p=json.loads(path.read_text(encoding='utf-8'))
        oldhash=fingerprint(p)
        backup=history/(ident+'-'+oldhash+'.json')
        if not backup.exists():backup.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
        for k in ('neutral','medium'):
            if k in d:p[k]=d[k]
        p['review_status']=d['status']
        p['agent_review_note']=d['note']
        p['review_flags']={k:fidelity_flags(p['source']['text'],p[k]) for k in ('neutral','medium')}
        p['review']=dict(reviewer='Codex agent',reviewed_at=datetime.now(timezone.utc).isoformat(),
            scope='Meaning and style suitability relative to supplied source; not factual truth or independent source authentication.',
            content_sha256=fingerprint(p),decision_file=decisions_path.name)
        path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')

def export():
    pairs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(PAIR_DIR.glob('*.json')) if p.name!='manifest.json']
    groups={};approved=[]
    for p in pairs:
        group,split=p['source']['group'],p['source']['split']
        if group in groups and groups[group]!=split:raise ValueError('Source group crosses splits')
        groups[group]=split
        if eligible(p):approved.append(p)
    export_dir=ROOT/'data/style/reviewed';export_dir.mkdir(exist_ok=True)
    for split in ('train','validation','test'):
        with (export_dir/(split+'.jsonl')).open('w',encoding='utf-8') as f:
            for p in approved:
                if p['source']['split']!=split:continue
                for level,source,target in [(0,p['source']['text'],p['neutral']),(50,p['neutral'],p['medium']),(100,p['neutral'],p['source']['text'])]:
                    r=dict(source_id=p['source']['id'],group=p['source']['group'],source_url=p['source']['url'],
                        review_sha256=p['review']['content_sha256'],level=level,
                        messages=[dict(role='system',content='Rewrite the input at the requested Trump style level while preserving its meaning.'),
                        dict(role='user',content=json.dumps(dict(text=source,intensity=level))),dict(role='assistant',content=target)])
                    f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=dict(total_pairs=len(pairs),statuses=dict(Counter(p['review_status'] for p in pairs)),
        approved_pairs=len(approved),approved_examples=3*len(approved),
        approved_pairs_by_split=dict(Counter(p['source']['split'] for p in approved)),
        invalidated_reviews=sum(p.get('review_status')=='approved_by_agent' and not eligible(p) for p in pairs),
        limitations=['Agent-reviewed, not independently human-reviewed or fact-checked.',
        'Source authorship and source content rights are not established by this review.',
        'Intensity labels remain ordinal editorial targets, not calibrated measurements.',
        'Small chronological holdouts are not representative of every topic or era. Review/export does not itself train a model.'])
    (export_dir/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (PAIR_DIR/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--decisions');args=parser.parse_args()
    if args.decisions:apply(ROOT/args.decisions)
    export()
