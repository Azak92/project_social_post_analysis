"""Download official weights at an immutable recorded revision into project work/."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'work/training'
os.environ['HF_HOME']=str(DEST/'hf-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from huggingface_hub import snapshot_download

source=json.loads((ROOT/'training/model-source.json').read_text())
DEST.mkdir(parents=True,exist_ok=True)
(DEST/'model-source.json').write_text(json.dumps(source,indent=2))
snapshot_download(source['repo'],revision=source['revision'],local_dir=DEST/'model',
                  allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','vocab*','merges*','LICENSE*','README.md'])
print('Pinned model downloaded to',DEST/'model')
