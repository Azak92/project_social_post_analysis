"""Two-epoch five-level pilot; fresh base, grouped data, fixed before/after cases."""
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WORK=ROOT/'work/training'
RUN=WORK/'five-level-pilot-v2'
os.environ.setdefault('HF_HOME',str(WORK/'hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
SYSTEM='Rewrite the input at the requested Trump style level while preserving its meaning.'

from training.five_level_data import load_reviewed, messages

def prompt_ids(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    return ids['input_ids'] if isinstance(ids,Mapping) else ids

def encode(tokenizer,row):
    prefix=prompt_ids(tokenizer,row['messages'][:-1])
    answer=tokenizer.encode(row['messages'][-1]['content'],add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
    if len(prefix)+len(answer)>512:raise ValueError(f"Overlength {row['source_id']} {row['level']}: {len(prefix)+len(answer)} tokens")
    return prefix+answer,[-100]*len(prefix)+answer

def main():
    import torch
    import transformers,peft
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from lab.evaluation_cases import cases,LEVELS
    if RUN.exists():raise RuntimeError('Run folder already exists; preserve it and choose a new run name in the script.')
    rows=load_reviewed()
    tokenizer=AutoTokenizer.from_pretrained(WORK/'model',local_files_only=True)
    encoded={s:[encode(tokenizer,r) for r in rows[s]] for s in ('train','validation')}
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    if torch.cuda.mem_get_info()[0]<7*1024**3:raise RuntimeError('Less than 7 GiB free; unload the inference model first.')
    RUN.mkdir(parents=True)
    config=dict(model=json.loads((ROOT/'training/model-source.json').read_text()),seed=42,epochs=2,
        rank=8,alpha=16,learning_rate=1e-4,microbatch=1,gradient_accumulation=8,max_tokens=512,
        train_rows=len(rows['train']),validation_rows=len(rows['validation']),test_rows_reserved=27,
        max_actual_tokens=max(len(x) for values in encoded.values() for x,_ in values),
        use_scope='User authorized local non-commercial archive training; no new license grant inferred.',
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((ROOT/'data/style/five-level-v2/groups.json').read_bytes()).hexdigest())
    (RUN/'implementation.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    (RUN/'data-loader.py').write_text((ROOT/'training/five_level_data.py').read_text(),encoding='utf-8')
    (RUN/'groups.json').write_bytes((ROOT/'data/style/five-level-v2/groups.json').read_bytes())
    (RUN/'config.json').write_text(json.dumps(config,indent=2))
    for s in ('train','validation'):(RUN/f'{s}-snapshot.json').write_text(json.dumps(rows[s],ensure_ascii=False),encoding='utf-8')
    fixed=cases();(RUN/'cases.json').write_text(json.dumps(fixed,ensure_ascii=False),encoding='utf-8')
    torch.manual_seed(42);random.seed(42);started=time.monotonic()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(WORK/'model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    targets=[n for n,_ in model.named_modules() if 'language_model' in n and n.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]
    if not targets:raise RuntimeError('Missing language adapter targets')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    trainable=[p for n,p in model.named_parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    def validation():
        model.eval();total=0;tokens=0
        with torch.no_grad():
            for ids,labels in encoded['validation']:
                x=torch.tensor([ids],device='cuda');y=torch.tensor([labels],device='cuda')
                loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
                n=sum(t!=-100 for t in labels[1:]);total+=loss.item()*n;tokens+=n
        return total/tokens
    def evaluate(mode):
        model.eval();model.gradient_checkpointing_disable()
        dest=RUN/mode;dest.mkdir()
        for case in fixed:
            for level in LEVELS:
                ids=prompt_ids(tokenizer,messages(case['text'],level))
                x=torch.tensor([ids],device='cuda')
                t=time.monotonic()
                with torch.no_grad():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
                tail=out[0,len(ids):].tolist()
                record=dict(case=case,level=level,mode=mode,text=tokenizer.decode(tail,skip_special_tokens=True).strip(),elapsed_seconds=round(time.monotonic()-t,3),length_limited=len(tail)>=160 and tail[-1]!=tokenizer.eos_token_id)
                (dest/f'{case["id"]}-{level}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'Evaluation {mode}: {case["id"]}',flush=True)
    before=validation();print(f'Validation before: {before:.5f}',flush=True)
    evaluate('before')
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    optimizer=torch.optim.AdamW(trainable,lr=1e-4)
    losses=[]
    for epoch in range(2):
        order=list(range(len(encoded['train'])));random.shuffle(order)
        torch.cuda.reset_peak_memory_stats()
        for start in range(0,len(order),8):
            batch=order[start:start+8];optimizer.zero_grad(set_to_none=True);mean=0
            for i in batch:
                ids,labels=encoded['train'][i];x=torch.tensor([ids],device='cuda');y=torch.tensor([labels],device='cuda')
                loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError('Non-finite train loss')
                mean+=loss.item()/len(batch);(loss/len(batch)).backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
            if not torch.isfinite(norm) or norm.item()==0:raise RuntimeError('Invalid gradients')
            optimizer.step();losses.append(mean);print(f'Train step {len(losses)}: loss {mean:.4f}',flush=True)
    peak=torch.cuda.max_memory_reserved()
    model.save_pretrained(RUN/'adapter');tokenizer.save_pretrained(RUN/'adapter')
    after=validation()
    report=dict(validation_loss_before=before,validation_loss_after=after,training_step_losses=losses,
        optimizer_steps=len(losses),trainable_parameters=sum(p.numel() for p in trainable),peak_reserved_bytes=peak,
        note='Two fixed epochs; validation loss is not meaning preservation or style quality. Test split untouched by optimization and evaluation.')
    (RUN/'report.json').write_text(json.dumps(report,indent=2))
    print(f'Validation after: {after:.5f}',flush=True)
    del optimizer
    evaluate('after')
    report['elapsed_seconds']=round(time.monotonic()-started,2);report['complete']=True
    (RUN/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
