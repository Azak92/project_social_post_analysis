# Training readiness and evaluation — 23 October 2026

## Current status update

The separate 2B environment is now installed and verified. A ten-step BF16 LoRA smoke test using four original, non-archive fixtures passed; adapter reload and a 512-token backward pass also passed. See `training/README.md`, `training/verified-run.json`, and `training/ARCHIVE_USE_REVIEW.md`. The earlier readiness observations below are historical; a subsequent user-authorized archive pilot has now completed. It reduced validation loss but lost slider sensitivity and has not been deployed; see training/archive-pilot-report.json.

## Original decision

Measure the existing 9B model before training. Do not equate strong stylistic imitation with a good rewrite: preservation of meaning is the first gate. No training, environment migration or new model download has been performed in this readiness check.

## Machine and environment observed

- NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB VRAM (12 GB class), driver 610.88.
- Python environment: PyTorch 2.9.1+cu128, CUDA available; Transformers 4.57.3. PEFT, TRL and bitsandbytes are absent. WSL is not installed.
- The existing Ollama model is a quantized GGUF inference model. A training experiment needs matching Transformers-format weights and a separate environment; it is not an in-place edit to that GGUF file.
- The GPU was using approximately 10 GB during the initial check. After the evaluation and explicit Ollama unload, GPU usage was 3,534 MiB with 8,464 MiB free (about 8.3 GiB). Total capacity is not available training memory. Recheck free VRAM after unloading Ollama; do not terminate other applications without identifying them.

## Proposed model and configuration

The initial hypothesis is **Qwen3.5-4B with BF16 LoRA**, not 9B QLoRA. The current Unsloth guide lists about 10 GB for 4B BF16 LoRA, about 22 GB for 9B, and discourages Qwen3.5 4-bit QLoRA because of quantization differences. These are vendor estimates, not measurements on this PC. With the measured 8.3 GiB free, do not attempt the approximately 10 GB 4B configuration yet. The immediate smoke-test candidate is 2B; retain 4B as the preferred follow-on experiment only if more headroom becomes available. A 4B run may still not fit alongside Windows and other GPU applications. Qwen3.5-2B (listed around 5 GB) is the fallback for a pipeline experiment; its quality must be measured rather than assumed.

Use an isolated training environment with Transformers v5 and compatible CUDA/PyTorch/Unsloth dependencies. Native Windows is documented as supported; WSL is an alternative, not a prerequisite we should install silently. Pin the working versions after a successful smoke test. Do not upgrade the website's Python environment in place.

Provisional smoke-test settings: text-only supervision, frozen vision components, language attention projection adapters, rank 8 / alpha 16, sequence length 512, microbatch 1, gradient accumulation 8, gradient checkpointing, learning rate 1e-4, fixed seed 42, assistant-response-only loss. Verify module names and attention masks with the selected model implementation. Measure a forward/backward pass and then 10 optimizer steps; inspect peak VRAM, finite loss, adapter gradients and sample outputs. These settings are proposals, not a tested recipe. Reject or revise rather than silently truncating long examples. Keep reasoning disabled consistently with the inference baseline and use the model's correct chat template.

After the smoke test, start with one epoch on approved training pairs, selecting checkpoints on validation only. Compare the *same* 4B model before and after training as well as the current 9B reference; otherwise parameter-count differences confound the comparison. Quantized export back to Ollama needs its own before/after quality check. Save the adapter separately and preserve the unchanged 9B model.

## Data readiness

75 accepted source pairs currently yield 168 training, 30 validation and 27 test messages. Three versions of one source are correlated, not three independent examples. Review decisions and source/content fingerprints are retained; modified or unreviewed records cannot enter exports. This is enough for a pipeline test, not evidence of broad generalization.

The next expansion target is 300–500 reviewed source pairs, balancing ordinary topics, political criticism, praise, requests, uncertainty and negative statements. Preserve author stance; neutral delivery is not a factual correction. Explicitly review intermediate targets at 25 and 75 in a subset before deciding whether interpolation from 0/50/100 is sufficient. Avoid making every high-style example political or hostile. Preserve grouped chronological holdouts and keep evaluation prompts out of training and example retrieval.

## Source-use check

The Qwen 9B model card declares Apache-2.0. Verify the exact selected 4B checkpoint and its license before downloading for training. The pinned Truth Social archive's local NOTICE declares `license: other` and explicitly does not grant rights to underlying posts; it asks users to verify upstream terms before training. This check therefore does **not** establish training permission. No fine-tuning on those posts has begun. Resolving source-use terms remains a readiness item; an alternative is a separately permissioned corpus. Source links/checksums establish provenance consistency, not authenticity of every claim or personal authorship.

## Fixed development evaluation

`lab/evaluation_cases.py` defines 30 original inputs, with explicit meaning invariants. Each is run at 0/25/50/75/100 in prompt-only and archive modes: 300 outputs. `python -m lab.evaluate` saves results under ignored `data/evaluation/baseline-v1/` and resumes completed records. Its configuration captures model digest, runtime version, cases and implementation hash; a changed configuration requires a new run name. Use `--run` for a different baseline. Results are visible at `/evaluation`.

The current baseline measures the entire existing archive prompt, including its instructions and nickname rule; it does not isolate retrieval alone. One seed and one output per condition are insufficient to establish statistical superiority. Latencies include model loading when applicable and are not a controlled speed benchmark.

Review rubric:

- Meaning: 2 intact, 1 minor shift, 0 changed/omitted claim, entity, quantity, attribution, certainty or condition.
- Fluency: 2 natural, 1 awkward, 0 broken.
- Observed style: 0 ordinary wording, 1 light distinctive phrasing, 2 several patterns, 3 strong cadence/vocabulary, 4 highly characteristic. Score independently of requested level; capitals alone are insufficient.
- Separately note unwanted politics, copied archive claims, personal identity claims, excessive length, and instruction injection.

For each input, compare all five observed style ratings to check whether style increases consistently. Prefer blinded, shuffled mode labels for a later independent rating round. Do not select a checkpoint on the test set. Because this fixed set is used to diagnose failures, call it a development benchmark; reserve a new unseen test after prompt/training decisions. Initial spot-review findings are diagnostic examples, not a scored accuracy estimate for all 300 outputs.

## Sources checked

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://unsloth.ai/docs/get-started/install/windows-installation
- Local pinned archive: `data/raw/truth/dde19cab0f3c5d783b9bc863700ecdd5a47d605c/NOTICE.md`


## Second, five-level pilot (current result)

Reviewed 30 content groups at all five levels: 150 targets, with plain and styled input views producing 240 training and 60 validation records. `/five-levels` displays the reviewed examples; `/training?run=v2` displays all 300 controlled before/after outputs and meaning-review findings. The original experiment remains at `/training?run=v1`.

The fresh 2B rank-8 adapter completed two fixed epochs / 60 optimizer steps. Validation loss fell from 0.9751 to 0.7748 (20.54%). Nevertheless, identical wording across all five levels increased from 5/30 to 16/30 cases; exact input copies rose from 73/150 to 109/150. Length-limited outputs fell from seven to zero. These metrics diagnose behavior, not semantic accuracy.

The assistant reviewed all 150 after-training outputs: five clear omissions/role/attribution errors and three ambiguous completion-to-intent shifts. Low-level nickname removal works on one case, but high-level style transfer is generally weak. The adapter was saved and reload-verified, but is not deployed. The live slider remains the existing 9B model. Twenty-five tests pass. Details and limitations are in `training/FIVE_LEVEL_PILOT.md` and `training/five-level-pilot-report.json`; run artifacts remain local under `work/training/five-level-pilot-v2`.

Next experiment: first test whether a wider adapter can deliberately learn a small, balanced five-level fixture. The current adapter touches only q/v projections in six full-attention layers. An expanded set of language layers is a hypothesis to test, not an established cure. Check memory and training-fit style control before spending another run on more data. Then add more varied, independently reviewed examples and new unseen evaluation cases. Do not judge success from loss or copying.
