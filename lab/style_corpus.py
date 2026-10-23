"""Prepare auditable style candidates, not a training-ready or author-verified set.

Run: python -m lab.style_corpus
Raw posts are untouched. Outputs stay under the Git-ignored data directory.
"""
import hashlib
import json
import re
from collections import Counter
from .data import ROOT, connect

PHRASES = ['dumocrat', 'crooked', 'sleepy', 'fake news', 'witch hunt',
           'radical left', 'thank you for your attention', 'tremendous',
           'beautiful', 'complete and total endorsement']

def exclusion(row):
    if row['platform'] == 'truth' and row['post_type'] != 'original':
        return 'quote_or_repost'
    if row['platform'] == 'twitter' and row['post_type'] == 'repost':
        return 'repost'
    text = row['text']
    if not text.strip():
        return 'empty'
    # Retain a separate lane for quoted material, even if source calls it original.
    if text.lstrip().startswith(('"', '“', '@')):
        return 'possible_quote_or_reply'
    without_links = re.sub(r'https?://\S+', '', text)
    if len(without_links.strip()) < len(text.strip()) * 0.5:
        return 'mostly_link'
    count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", without_links))
    if not 12 <= count <= 300:
        return 'outside_pilot_length'
    return None

def main():
    dest = ROOT / 'data' / 'style'
    dest.mkdir(parents=True, exist_ok=True)
    counts, phrase_all, phrase_candidates = Counter(), Counter(), Counter()
    candidates, excluded, seen = [], [], {}
    with connect() as db:
        rows = [dict(r) for r in db.execute('SELECT * FROM posts ORDER BY created_at,platform,id')]
    for row in rows:
        for phrase in PHRASES:
            phrase_all[phrase] += int(phrase in row['text'].lower())
        reason = exclusion(row)
        key = hashlib.sha256(re.sub(r'\s+', ' ', row['text']).strip().lower().encode()).hexdigest()
        if not reason and key in seen:
            reason = 'exact_text_duplicate'
        if reason:
            counts[reason] += 1
            excluded.append(dict(platform=row['platform'],id=row['id'],reason=reason))
            continue
        seen[key] = (row['platform'],row['id'])
        lane = 'truth_original_unverified_author' if row['platform']=='truth' else 'twitter_unknown_type_and_author'
        counts[lane] += 1
        for phrase in PHRASES:
            phrase_candidates[phrase] += int(phrase in row['text'].lower())
        candidates.append(dict(platform=row['platform'],id=row['id'],text=row['text'],
            created_at=row['created_at'],url=row['url'],source=row['source'],revision=row['revision'],
            text_sha256=key,lane=lane,status='candidate_not_reviewed'))
    report = dict(source_posts=len(rows),candidate_posts=len(candidates),counts=dict(counts),
        phrase_document_counts_all=dict(phrase_all),phrase_document_counts_candidates=dict(phrase_candidates),
        limitations=[
            'Original-post labels do not prove personal authorship.',
            'Quotation and link filters are heuristics; embedded quoted material remains possible.',
            'Exact duplicates removed; near-duplicate and campaign-template grouping still required.',
            'No train/evaluation split yet. Group near duplicates before a chronological split.',
            'No neutral pairs, intensity labels, training run, or model-weight changes made.',
            'Review source terms and dataset notice before training or redistribution.',
            'Phrase counts are substring matches per post, not occurrence counts or proof of unique style.'
        ])
    for filename, records in [('candidates.jsonl',candidates),('exclusions.jsonl',excluded)]:
        with (dest/filename).open('w',encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (dest/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
