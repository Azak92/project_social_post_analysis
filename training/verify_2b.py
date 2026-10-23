"""Reload the disposable adapter and check a synthetic 512-token backward pass."""
import json
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from smoke_2b import WORK

tokenizer=AutoTokenizer.from_pretrained(WORK/'model',local_files_only=True)
base=Qwen3_5ForConditionalGeneration.from_pretrained(WORK/'model',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
model=PeftModel.from_pretrained(base,WORK/'smoke-adapter',is_trainable=True)
model.eval()
messages=[{'role':'user','content':'Rewrite plainly: Please return the green stapler after the workshop.'}]
inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
    enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda')
with torch.no_grad():output=model.generate(**inputs,max_new_tokens=40,do_sample=False)
sample=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
assert sample.strip()
model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.enable_input_require_grads()
# Mechanical tensor stress only: repeated original filler is not a training corpus.
tokens=tokenizer.encode('A ceramic cup sits beside a wooden tray. ',add_special_tokens=False)
ids=(tokens*512)[:512]
x=torch.tensor([ids],device='cuda');labels=x.clone();labels[:,:496]=-100
torch.cuda.reset_peak_memory_stats()
loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=labels,use_cache=False).loss
assert torch.isfinite(loss)
loss.backward()
grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
report=dict(adapter_reload_passed=True,sample_output=sample,stress_sequence_tokens=512,
    stress_loss=loss.item(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    note='Synthetic capacity test; no optimizer step, no archive data, no quality conclusion.')
(WORK/'verification-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
