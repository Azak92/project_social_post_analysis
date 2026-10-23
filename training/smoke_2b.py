"""Local BF16 LoRA plumbing test using original fixtures only, never archive data."""
import json
import os
from collections.abc import Mapping
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/training'
os.environ.setdefault('HF_HOME',str(WORK/'hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')

# Original mechanical fixtures, separate from both archive and evaluation inputs.
FIXTURES=[
    ('Please return the green stapler after the workshop.',0,'Please return the green stapler after the workshop.'),
    ('Our pottery class made six bowls this morning.',50,'Six bowls from our pottery class this morning. A productive session!'),
    ('The bakery opens at seven on Monday.',0,'The bakery opens at seven on Monday.'),
    ('I liked the new mural beside the station.',50,'That new mural beside the station looks great. I liked it!'),
]

def encode_example(tokenizer,text,level,target):
    messages=[{'role':'system','content':'Rewrite the input at the requested style intensity. Preserve meaning.'},
              {'role':'user','content':json.dumps({'text':text,'intensity':level})}]
    prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if isinstance(prefix,Mapping):prefix=prefix['input_ids']
    target_ids=tokenizer.encode(target,add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
    ids=prefix+target_ids
    if len(ids)>512:raise ValueError('Fixture exceeds sequence budget; no silent truncation.')
    return ids,[-100]*len(prefix)+target_ids

def main():
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    WORK.mkdir(parents=True,exist_ok=True)
    source=json.loads((WORK/'model-source.json').read_text())
    path=WORK/'model'
    if not (path/'config.json').exists():raise RuntimeError('Download the pinned model first.')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 is required.')
    torch.manual_seed(42)
    free,total=torch.cuda.mem_get_info()
    if free<6*1024**3:raise RuntimeError('Less than 6 GiB free. Release the inference model and retry.')
    report=dict(model=source,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                gpu=torch.cuda.get_device_name(),free_bytes_before=free,steps=10,sequence_limit=512,
                microbatch=1,gradient_accumulation=4,learning_rate=1e-4,rank=8,alpha=16,
                data='Four original plumbing fixtures; no archive or evaluation inputs.',losses=[])
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    samples=[encode_example(tokenizer,*r) for r in FIXTURES]
    report['token_lengths']=[len(ids) for ids,_ in samples]
    report['target_token_counts']=[sum(x!=-100 for x in labels) for _,labels in samples]
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,
            trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    targets=[name for name,_ in model.named_modules() if 'language_model' in name and name.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]
    if not targets:raise RuntimeError('No verified language attention targets found.')
    report['target_modules']=targets
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n,p in trainable)
    report['trainable_parameters']=sum(p.numel() for _,p in trainable)
    report['total_parameters']=sum(p.numel() for p in model.parameters())
    before={n:p.detach().cpu().clone() for n,p in trainable}
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=1e-4)
    model.train()
    for step in range(10):
        optimizer.zero_grad(set_to_none=True);loss_value=0
        for ids,labels in samples:
            x=torch.tensor([ids],device='cuda');y=torch.tensor([labels],device='cuda')
            loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss.')
            loss_value+=loss.item()/len(samples)
            (loss/len(samples)).backward()
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
        if not torch.isfinite(norm) or norm.item()==0:raise RuntimeError('Missing or non-finite adapter gradients.')
        optimizer.step();report['losses'].append(loss_value)
        print(f'Step {step+1}/10 loss={loss_value:.4f}',flush=True)
    assert any(not torch.equal(before[n],p.detach().cpu()) for n,p in trainable)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    report['passed']=True
    report['meaning']='Training plumbing only; not a trained Trump-style model or quality improvement claim.'
    out=WORK/'smoke-adapter';model.save_pretrained(out);tokenizer.save_pretrained(out)
    (WORK/'smoke-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
