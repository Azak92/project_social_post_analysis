"""Reproducible local archive: immutable sources and idempotent imports."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'lab.sqlite3'
DATASET = 'Cameronk199/donald-trump-truth-social-posts'
PINNED = 'dde19cab0f3c5d783b9bc863700ecdd5a47d605c'

@contextmanager
def connect(path=DB):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS posts (
        platform TEXT, id TEXT, created_at TEXT, timezone TEXT,
        text TEXT, raw_text TEXT, url TEXT, post_type TEXT,
        source TEXT, revision TEXT, PRIMARY KEY(platform,id))''')
    db.execute('CREATE INDEX IF NOT EXISTS date_index ON posts(created_at)')
    try:
        with db:
            yield db
    finally:
        db.close()

def normalize(text):
    # Raw text remains available; repairs are explicitly recorded by differences.
    import ftfy
    return ftfy.fix_text(text or '').strip()

def upsert(db, row):
    db.execute('''INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(platform,id) DO UPDATE SET
        created_at=excluded.created_at, timezone=excluded.timezone,
        text=excluded.text, raw_text=excluded.raw_text, url=excluded.url,
        post_type=excluded.post_type, source=excluded.source, revision=excluded.revision''', row)

def import_twitter(db, path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count = 0
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            match = re.search(r'/status/(\d+)', row['Tweet URL'])
            if not match:
                raise ValueError(f'Missing post ID in {path.name}, row {count+2}')
            stamp = datetime.strptime(row['Time'], '%Y-%m-%d %H:%M').isoformat()
            raw = row['Tweet Text']
            upsert(db, ('twitter', match[1], stamp, 'unknown', normalize(raw), raw,
                        row['Tweet URL'], 'repost' if raw.lstrip().startswith('RT @') else 'unknown',
                        path.name, digest))
            count += 1
    return count

def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()

def download_truth(revision=PINNED):
    if revision == 'latest':
        revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Use an immutable 40-character commit SHA or latest')
    dest = ROOT / 'data' / 'raw' / 'truth' / revision
    dest.mkdir(parents=True, exist_ok=True)
    base = f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/'
    checksums = fetch(base + 'checksums.sha256')
    expected = {line.split(maxsplit=1)[1].lstrip('*'): line.split()[0]
                for line in checksums.decode().splitlines() if line.strip()}
    for name in ['NOTICE.md', 'README.md', 'manifest.json', 'data/full_archive.posts.v1.jsonl.gz']:
        target = dest / Path(name).name
        content = target.read_bytes() if target.exists() else fetch(base + name)
        if hashlib.sha256(content).hexdigest() != expected[name]:
            raise ValueError(f'Checksum mismatch: {name}')
        target.write_bytes(content)
    (dest / 'checksums.sha256').write_bytes(checksums)
    (dest / 'receipt.json').write_text(json.dumps(dict(dataset=DATASET, revision=revision,
        retrieved_at=datetime.now(timezone.utc).isoformat()), indent=2), encoding='utf-8')
    return dest, revision

def import_truth(db, path, revision):
    count = 0
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ident = row.get('truth_id')
            if not ident:
                raise ValueError('Missing Truth Social ID')
            raw = row.get('content_text') or ''
            upsert(db, ('truth', str(ident), row['created_at_utc'], 'UTC', normalize(raw), raw,
                row.get('truth_url') or row['source_url'], row['post_type'], row['source_url'], revision))
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth', action='store_true')
    parser.add_argument('--revision', default=PINNED)
    args = parser.parse_args()
    downloaded = download_truth(args.revision) if args.truth else None
    with connect() as db:
        counts = {p.name: import_twitter(db, p) for p in sorted((ROOT / 'data').glob('realDonaldTrump*.csv'))}
        if downloaded:
            dest, revision = downloaded
            counts['truth'] = import_truth(db, dest / 'full_archive.posts.v1.jsonl.gz', revision)
        counts['unique_posts'] = db.execute('SELECT count(*) FROM posts').fetchone()[0]
    print(json.dumps(counts, indent=2))

if __name__ == '__main__':
    main()
