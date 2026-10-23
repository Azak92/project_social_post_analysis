"""Resumable fixed evaluation of the live rewrite implementation; no training."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .app import RewriteInput, rewrite, ollama
from .data import ROOT
from .evaluation_cases import cases, LEVELS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='qwen3.5:9b-q4_K_M')
    parser.add_argument('--run',default='baseline-v1')
    args=parser.parse_args()
    if not args.run.replace('-','').replace('_','').isalnum():parser.error('Use a simple run name.')
    dest=ROOT/'data/evaluation'/args.run;dest.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    for file in ['lab/app.py','lab/archive_style.py','lab/evaluation_cases.py','lab/evaluate.py']:
        source_hash.update((ROOT/file).read_bytes())
    model=next(m for m in ollama('tags')['models'] if m['name']==args.model)
    config=dict(model=args.model,digest=model['digest'],implementation_sha256=source_hash.hexdigest(),
                levels=list(LEVELS),modes=['baseline','archive'],cases=cases(),runtime=ollama('version'))
    manifest=dest/'config.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=config:
        raise ValueError('Run configuration changed. Choose a new run name; do not mix baselines.')
    manifest.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    completed=[]
    for case in cases():
        for level in LEVELS:
            # Alternate ordering to spread cold-start/order effects across modes.
            modes=config['modes'] if level%50==0 else list(reversed(config['modes']))
            for mode in modes:
                path=dest/f"{case['id']}-{level}-{mode}.json"
                if path.exists():
                    completed.append(json.loads(path.read_text(encoding='utf-8')));continue
                result=rewrite(RewriteInput(text=case['text'],model=args.model,intensity=level,mode=mode))
                record=dict(case=case,level=level,mode=mode,result=result,review_status='unreviewed',
                            generated_at=datetime.now(timezone.utc).isoformat())
                path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                completed.append(record)
                print(f'{len(completed)}/300 {case["id"]} {level} {mode}',flush=True)
    summary=dict(outputs=len(completed),cases=len(cases()),model=args.model,
        flagged_by_mode=dict(Counter(r['mode'] for r in completed if r['result']['review_flags'])),
        note='Flags are review cues, not semantic accuracy. Style and meaning require separate review. No training.')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))

if __name__=='__main__':main()
