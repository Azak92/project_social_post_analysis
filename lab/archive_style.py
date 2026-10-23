"""Source-linked lexical retrieval and leakage-aware corpus partitions."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from .data import ROOT

STOP = set('the a an is are was were be been it its i we you he she they them our your their to of and or in on for with at by from this that have has had will would do did but as very so'.split())

def tokens(text):
    return re.findall(r"[a-z]+",text.lower())

def shingles(text):
    words=tokens(re.sub(r'https?://\S+|\d+',' ',text))
    return set(zip(words,words[1:],words[2:],words[3:]))

def category(text):
    words=set(tokens(text))
    if words & {'democrats','democrat','dumocrats','dumocrat','republicans','republican','election','biden','crooked','radical','rino','rinos','desanctimonious','desantis','senator','congress','secretary','border','borders','obama','kamala','hillary','pelosi','schumer','mccarthy','mitch','mcconnell','medicare','political','president','woke','transgender'}: return 'politics'
    if words & {'congratulations','congratulate','proud','winner','winning','won','excellent','team'}: return 'praise'
    if words & {'thank','thanks','appreciate','honor'}: return 'gratitude'
    if words & {'fake','terrible','disgrace','bad','wrong','failed'}: return 'criticism'
    if words & {'economy','jobs','business','trade','dollars','money','tax'}: return 'business'
    return 'general'

@lru_cache(maxsize=1)
def corpus():
    path=ROOT/'data/style/reference-index.json'
    if not path.exists():
        raise FileNotFoundError('Run python -m lab.archive_style to build archive references.')
    rows=json.loads(path.read_text(encoding='utf-8'))
    df=Counter(w for r in rows if r['split']=='train' for w in set(tokens(r['text']))-STOP)
    return rows,df

def select_examples(text, limit=3):
    rows,df=corpus(); query=set(tokens(text))-STOP; kind=category(text)
    ranked=[]
    query_shingles=shingles(text)
    for row in rows:
        if row['split']!='train': continue
        if len(row['text'].split())>60:continue
        words=set(tokens(row['text']))-STOP
        # Neutral topics do not receive political attack examples.
        if kind!='politics' and row['category']=='politics': continue
        rs=shingles(row['text'])
        if query_shingles and len(query_shingles & rs)/max(1,len(query_shingles|rs))>.65: continue
        score=sum(math.log(1+len(rows)/(1+df[w])) for w in query&words)
        score = score / math.sqrt(max(1,len(words)/12))
        score += 4*(row['category']==kind)
        if query & {'democrats','dumocrats','democrat','dumocrat'} and words & {'dumocrats','dumocrat'}:score+=12
        score += min(2, row['style_signals']/3)
        ranked.append((score,row))
    ranked.sort(key=lambda pair:(-pair[0],pair[1]['id']))
    chosen=[];groups=set()
    for score,row in ranked:
        if row['group'] in groups: continue
        chosen.append(dict(row,retrieval_score=round(score,2)))
        groups.add(row['group'])
        if len(chosen)==limit: break
    return chosen

def build():
    source=ROOT/'data/style/candidates.jsonl'
    rows=[]
    for line in source.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        # First experiment: modern original-post lane, short text without links/signatures.
        if r['platform']!='truth' or not 12<=len(r['text'].split())<=100: continue
        if re.search(r'https?://|@|complete and total endorsement',r['text'],re.I):continue
        r['original_text']=r['text']
        r['text']=re.sub(r'\s*(?:President\s+)?(?:DONALD\s+J\.?\s*TRUMP|DJT)[.!]?\s*$', '', r['text'],flags=re.I).strip()
        r['category']=category(r['text'])
        r['style_signals']=len(re.findall(r'\b[A-Z]{3,}\b',r['text']))+r['text'].count('!')+len(re.findall(r'\b(tremendous|beautiful|dumocrats|crooked|fake|sad|great|incredible)\b',r['text'],re.I))
        rows.append(r)
    # Join near-duplicates before assigning splits; all members follow the newest date.
    parent=list(range(len(rows))); index=defaultdict(list); sets=[]
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        ss=shingles(row['text']); overlaps=Counter(j for s in ss for j in index[s])
        for j,intersection in overlaps.items():
            if intersection/max(1,len(ss)+len(sets[j])-intersection)>=.65:
                parent[find(i)]=find(j)
        sets.append(ss)
        for s in ss:index[s].append(i)
    dates={}; ids=defaultdict(list)
    for i,row in enumerate(rows):
        root=find(i);dates[root]=max(dates.get(root,''),row['created_at']);ids[root].append(row['id'])
    for i,row in enumerate(rows):
        root=find(i);date=dates[root]
        row['group']=hashlib.sha256('|'.join(sorted(ids[root])).encode()).hexdigest()[:16]
        row['split']='train' if date<'2026-06-01' else 'validation' if date<'2026-07-01' else 'test'
    dest=ROOT/'data/style/reference-index.json';dest.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    report=dict(posts=len(rows),groups=len(ids),splits=dict(Counter(r['split'] for r in rows)),
        policy='Truth originals, 12–100 words, no links/handles/signatures; four-word shingle Jaccard >=0.65 groups; group newest date sets split. Retrieval uses train only. Heuristic grouping is not exhaustive.')
    (dest.parent/'reference-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
