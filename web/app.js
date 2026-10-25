const $ = id => document.getElementById(id);
const outputs = {baseline: $('baseline-output'), archive: $('archive-output')};
let busy = false;
async function api(url, body) {
  const response = await fetch(url, body ? {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  } : {});
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Please check your input and try again.');
  return data;
}
$('intensity').oninput = () => {
  $('value').textContent = $('intensity').value;
  $('intensity').setAttribute('aria-valuetext', $('intensity').value + ' out of 100');
};
$('rewrite-form').onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  const text = $('input').value.trim();
  if (!text) { $('error').textContent = 'Enter some text first.'; $('input').focus(); return; }
  const intensity = Number($('intensity').value);
  busy = true;
  $('error').textContent = '';
  for (const id of ['generate', 'input', 'intensity', 'topic', 'topic-generate']) $(id).disabled = true;
  $('generate').textContent = 'Generating…';
  $('status').textContent = 'Creating both versions. The first request may take a little longer.';
  for (const output of Object.values(outputs)) {
    output.className = 'output placeholder'; output.textContent = 'Waiting to generate…';
  }
  let completed = 0;
  try {
    const {models} = await api('/api/models');
    if (!models.length) throw Error('Start Open Language Lab, then try again.');
    const preferred = 'qwen3.5:9b-q4_K_M';
    const model = models.includes(preferred) ? preferred : models[0];
    // Sequential requests share one local GPU model.
    for (const mode of ['baseline', 'archive']) {
      outputs[mode].textContent = 'Generating…';
      try {
        const result = await api('/api/rewrite', {text, intensity, model, mode});
        outputs[mode].textContent = result.text;
        outputs[mode].className = 'output';
        completed++;
      } catch (error) {
        outputs[mode].textContent = 'Could not generate this version. ' + error.message;
        outputs[mode].className = 'output failed';
      }
    }
    $('status').textContent = completed === 2 ? '' : 'Generate again to retry both versions.';
  } catch (error) {
    $('error').textContent = error.message;
    $('status').textContent = '';
    for (const output of Object.values(outputs)) output.textContent = 'No result generated yet.';
  } finally {
    busy = false;
    for (const id of ['generate', 'input', 'intensity', 'topic', 'topic-generate']) $(id).disabled = false;
    $('generate').textContent = 'Generate';
  }
};


$('topic-form').onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  const topic = $('topic').value.trim();
  if (!topic) { $('topic-error').textContent = 'Enter a topic first.'; $('topic').focus(); return; }
  busy = true;
  for (const id of ['generate', 'input', 'intensity', 'topic', 'topic-generate']) $(id).disabled = true;
  $('topic-generate').textContent = 'Generating…';
  $('topic-error').textContent = '';
  $('topic-status').textContent = 'Finding archive examples and writing a new post…';
  $('topic-result').hidden = true;
  try {
    const {models} = await api('/api/models');
    if (!models.length) throw Error('Start Open Language Lab, then try again.');
    const preferred = 'qwen3.5:9b-q4_K_M';
    const model = models.includes(preferred) ? preferred : models[0];
    const result = await api('/api/topic', {topic, model});
    $('topic-output').textContent = result.text;
    $('topic-result').hidden = false;
  } catch (error) {
    $('topic-error').textContent = error.message;
  } finally {
    busy = false;
    $('topic-status').textContent = '';
    $('topic-generate').textContent = 'Generate post';
    for (const id of ['generate', 'input', 'intensity', 'topic', 'topic-generate']) $(id).disabled = false;
  }
};
