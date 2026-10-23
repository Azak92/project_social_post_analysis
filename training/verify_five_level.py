"""Reload the saved adapter and reproduce one fixed evaluation output."""
import json
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from five_level_pilot import ROOT, WORK, RUN, prompt_ids, messages

expected=json.loads((RUN/'after/forecast-100.json').read_text(encoding='utf-8'))
tokenizer=AutoTokenizer.from_pretrained(RUN/'adapter',local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(WORK/'model',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(model,RUN/'adapter');model.eval()
ids=prompt_ids(tokenizer,messages(expected['case']['text'],100))
x=torch.tensor([ids],device='cuda')
with torch.no_grad():
    out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=160,
        do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
text=tokenizer.decode(out[0,len(ids):],skip_special_tokens=True).strip()
report=dict(reload_passed=text==expected['text'],case='forecast',level=100,text=text)
(RUN/'reload-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
assert report['reload_passed'], 'Saved adapter output differs from training process'
print(json.dumps(report))
