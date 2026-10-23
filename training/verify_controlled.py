"""Verify a saved broader adapter reproduces its fixed transfer output."""
import json
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from controlled_wide_pilot import WORK,RUN,prompt_ids,messages

expected=json.loads((RUN/'after/transfer_payment-100.json').read_text(encoding='utf-8'))
tokenizer=AutoTokenizer.from_pretrained(RUN/'adapter',local_files_only=True)
base=Qwen3_5ForConditionalGeneration.from_pretrained(WORK/'model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(base,RUN/'adapter');model.eval()
ids=prompt_ids(tokenizer,messages(expected['case']['text'],100));x=torch.tensor([ids],device='cuda')
with torch.no_grad():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
text=tokenizer.decode(out[0,len(ids):],skip_special_tokens=True).strip()
result=dict(reload_passed=text==expected['text'],case='transfer_payment',level=100,text=text)
(RUN/'reload-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
assert result['reload_passed'],'Saved adapter differs from original run'
print(json.dumps(result))
