const $=id=>document.getElementById(id);let offset=0,total=0;
async function api(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Please check your input.');return d}
function report(e){$('error').textContent=e.message}
async function analyze(){ $('chars').textContent=`${$('input').value.length} / 4000`;if(!$('input').value.trim()){$('metrics').replaceChildren();return}const d=await api('/api/analyze',{text:$('input').value});$('metrics').replaceChildren();for(const [k,v] of Object.entries(d)){const box=document.createElement('div');box.className='metric';const n=document.createElement('b');n.textContent=v;const label=document.createElement('span');label.textContent=k.replaceAll('_',' ');box.append(n,label);$('metrics').append(box)}}
async function search(){const d=await api('/api/posts?'+new URLSearchParams({q:$('search').value,platform:$('platform').value,offset}));total=d.total;$('matches').textContent=`${total.toLocaleString()} matches`;$('posts').replaceChildren();for(const p of d.posts){const row=document.createElement('article');row.className='post';const meta=document.createElement('div');meta.className='meta';meta.textContent=`${p.platform==='truth'?'TRUTH SOCIAL':'TWITTER'} · ${p.post_type} · ${p.created_at} (${p.timezone})`;const text=document.createElement('p');text.textContent=p.text||'[No text: media-only post]';const link=document.createElement('a');if(/^https:\/\/(twitter\.com|truthsocial\.com|trumpstruth\.org)\//.test(p.url)){link.href=p.url;link.target='_blank';link.rel='noopener';link.textContent='View source ↗'}row.append(meta,text,link);$('posts').append(row)}if(!d.posts.length)$('posts').textContent='No matching posts.';$('prev').disabled=offset===0;$('next').disabled=offset+20>=total;$('page').textContent=total?`${offset+1}–${Math.min(offset+20,total)} of ${total.toLocaleString()}`:'0 results'}
$('find').onclick=()=>{offset=0;search().catch(report)};$('search').onkeydown=e=>{if(e.key==='Enter')$('find').click()};$('platform').onchange=()=>$('find').click();$('prev').onclick=()=>{offset=Math.max(0,offset-20);search().catch(report)};$('next').onclick=()=>{offset+=20;search().catch(report)};$('intensity').oninput=()=>$('value').textContent=$('intensity').value;let timer;$('input').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>analyze().catch(report),200)};
let hasResult=false;
async function transform(compare=false){
  const text=$('input').value, intensity=Number($('intensity').value), model=$('model').value, mode=$('mode').value;
  if(!text.trim()){report(Error('Enter a post first.'));return}
  $('error').textContent='';$('result').hidden=true;$('comparison').hidden=true;$('references').hidden=true;
  for(const id of ['rewrite','input','intensity','model','mode','compare','unload'])$(id).disabled=true;
  $('rewrite').textContent='Transforming…';
  $('modelstatus').textContent='Rewriting locally. The first request may take longer while the model loads.';
  try{
    const d=await api('/api/rewrite',{text,intensity,model,mode:compare?'archive':mode});
    $('output').textContent=d.text;$('result').hidden=false;hasResult=true;let review=$('reviewflags');if(!review){review=document.createElement('p');review.id='reviewflags';review.className='note';$('result').append(review)}review.textContent=d.review_flags.length?'Review: '+d.review_flags.join(' '):'No numeric, uncertainty-word or negation flags detected. This is a limited check, not a meaning guarantee.';
    $('reference-list').replaceChildren();for(const r of d.references){const p=document.createElement('p');p.textContent=r.text;const a=document.createElement('a');a.textContent='View source ↗';if(/^https:\/\/(truthsocial\.com|trumpstruth\.org)\//.test(r.url)){a.href=r.url;a.target='_blank';a.rel='noopener'}$('reference-list').append(p,a)}$('references').hidden=!d.references.length;
    if(compare){const b=await api('/api/rewrite',{text,intensity,model,mode:'baseline'});$('baseline-output').textContent=b.text;$('baseline-flags').textContent=b.review_flags.join(' ');$('comparison').hidden=false}
    $('modelstatus').textContent=`${d.mode==='archive'?'Archive examples':'Prompt only'} · ${d.model} · intensity ${d.intensity}/100 · ${d.elapsed_seconds}s · generated locally`;
  }catch(e){report(e);$('modelstatus').textContent='Could not complete the rewrite. You can try again.'}
  finally{for(const id of ['rewrite','input','intensity','model','mode','compare','unload'])$(id).disabled=false;$('rewrite').textContent='Transform post ↗';refreshRuntime().catch(report)}
}
$('rewrite').onclick=()=>transform(false);$('compare').onclick=()=>transform(true);
$('intensity').onchange=()=>{if(hasResult)$('rewrite').click()};
async function init(){const s=await api('/api/stats');$('total').textContent=s.groups.reduce((a,g)=>a+g.count,0).toLocaleString();$('coverage').textContent=s.groups.map(g=>`${g.platform==='truth'?'Truth Social':'Twitter'}: ${g.first.slice(0,10)} → ${g.last.slice(0,10)}`).join(' · ');await Promise.all([analyze(),search()]);const m=await api('/api/models');if(m.models.length){$('model').replaceChildren(...m.models.map(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o}));$('model').value=m.models.includes('qwen3.5:9b-q4_K_M')?'qwen3.5:9b-q4_K_M':m.models[0];$('rewrite').disabled=false;$('modelstatus').textContent='Runs locally. Transform once, then release the slider to regenerate. Check meaning before use.'}else{$('modelstatus').textContent='Model connection ready. Choose and install a local Ollama model to activate rewriting. Text analysis and archive search work now.'}}
async function refreshRuntime(){
  const d=await api('/api/runtime');
  const loaded=d.loaded.find(m=>m.name===$('model').value||m.model===$('model').value);
  $('runtime-status').textContent=!d.available?'Model service is offline. Use Open Language Lab to start it.':loaded?`Model in memory: ${(loaded.size_vram/1024**3).toFixed(1)} GiB on GPU. It unloads after five idle minutes. Residency does not mean active generation.`:'Selected model is unloaded. Its files remain on disk; the next rewrite loads it again.';
  $('unload').disabled=!loaded||$('input').disabled;
}
$('refresh-runtime').onclick=()=>refreshRuntime().catch(report);
$('model').onchange=()=>refreshRuntime().catch(report);
$('unload').onclick=async()=>{ $('unload').disabled=true;try{await api('/api/unload',{model:$('model').value});await refreshRuntime()}catch(e){report(e);$('unload').disabled=false} };
init().then(refreshRuntime).catch(report);
setInterval(()=>{if(!document.hidden)refreshRuntime().catch(()=>{})},30000);
