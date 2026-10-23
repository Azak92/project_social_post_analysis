# Isolated 2B training test

The environment is `work/train-2b-env`, separate from the website and its Python packages. Model weights, cache, logs and disposable smoke-test adapter stay in ignored `work/training/`.

This first test uses standard CUDA PyTorch, Transformers v5 and PEFT, without Unsloth, TRL, bitsandbytes or custom Triton kernels. BF16 keeps the frozen base model at 16-bit precision; LoRA updates only small attention adapters. Standard kernels may be slower than Unsloth, so vendor Unsloth memory/speed estimates do not describe this implementation. Measure the actual run.

`download_2b.py` uses the official Qwen/Qwen3.5-2B model and its recorded immutable revision. `smoke_2b.py` uses only four newly written fixtures, never the archive, reviewed pairs or evaluation prompts. It checks CUDA BF16, input/target masking, finite loss, nonzero adapter gradients, changing adapter weights, and peak GPU allocation during ten optimizer steps. Only assistant-answer tokens contribute to loss. The base and vision weights stay frozen; the vision encoder is unused for text inputs. The output adapter is disposable and is not connected to the slider.

Run from the project root with the environment's Python:

```
work\train-2b-env\Scripts\python.exe training\download_2b.py
work\train-2b-env\Scripts\python.exe training\smoke_2b.py
```

The source-use review is in `ARCHIVE_USE_REVIEW.md`. It distinguishes provenance, rights and personal authorship. It does not treat the archive's public availability or research purpose as a blanket training license. Archive fine-tuning is a separate decision from this mechanical environment test.


## Verified result

On 23 October 2026, the isolated environment used Python 3.13.1, PyTorch 2.9.1+cu128, Transformers 5.20.2, PEFT 0.21.2 and Accelerate 1.15.0. `requirements-lock.txt` captures installed versions; `setup.ps1` recreates the environment using the official CUDA wheel index before the lock file. Model revision and license are retained in `model-source.json` and the downloaded model directory.

Ten optimizer steps passed in 15.15 seconds including model load: four short original fixtures (58–65 tokens), microbatch 1, accumulation 4, rank 8 / alpha 16, learning rate 1e-4. There were 417,792 trainable parameters, limited to query/value adapters in the six full-attention language layers; linear-attention layers, embeddings, output head and vision weights stayed frozen. The loss changed from 1.2164 to 0.8189 on the same tiny fixtures. This is a mechanical fit check, not a generalization result.

Peak reserved GPU memory was 5,010,096,128 bytes (4.67 GiB). A separate adapter-reload and generation check passed. A 512-token synthetic forward/backward check also passed, reserving 6,387,924,992 bytes (5.95 GiB). That stress test made no optimizer update. Standard PyTorch reference kernels were used where optimized linear-attention/convolution extensions were absent. Peak CUDA allocator measurements exclude other applications and driver memory. Longer batches and different adapter targets need fresh measurements.

`verified-run.json` retains the smoke report, reload/capacity check and local weight checksum. Run `verify_2b.py` for the reload/capacity test. The test processes exit and release their GPU allocations. The website remains on the unchanged 9B model; this disposable adapter has not been deployed. Eighteen project tests pass.


## Authorized archive pilot

The user subsequently authorized local non-commercial training on the archive. `archive_pilot.py` implements a separate one-epoch experiment, starting from the official base model rather than the disposable smoke-test adapter. It checks both review fingerprints and exact exported messages, disallows cross-split groups and silent truncation, and snapshots the train/validation data plus model/configuration hashes into `work/training/archive-pilot-v1/`. Existing run directories are protected from overwrite.

Configuration: BF16, six language full-attention query/value adapters, rank 8 / alpha 16, learning rate 1e-4, microbatch 1, accumulation 8, seed 42, 512-token ceiling. The reviewed inputs fit within 261 tokens. One epoch uses 168 messages from 56 source groups, giving 21 optimizer steps. Thirty validation messages from ten source groups are evaluated before and after without gradient updates; the 27-message test split is reserved. Training loss averages per example; validation loss is weighted by supervised target tokens.

The controlled comparison uses the same brief training-format instruction, greedy decoding, a 160-token output limit, and the same 30 original evaluation inputs at all five levels, before and after the adapter update. It does not use retrieval, so comparisons against the existing 9B archive-assisted site are confounded by model size and prompt differences. Compare within this 2B experiment first. The full 300 saved outputs appear at `/training`. The 25/75 levels are interpolation checks because training labels are 0/50/100.

The website's live slider remains on the original 9B inference model. The pilot adapter is saved separately and has not been merged, quantized, published or deployed. Authorization records the intended scope, not a new underlying-content license; the source-use review remains available.


### Pilot outcome

The run completed in 372.33 seconds including 300 generated comparisons. Validation loss fell from 1.56751 to 1.37075 (12.55%). The CUDA allocator reserved a peak of 8,269,070,336 bytes (7.70 GiB). The saved adapter reloaded and exactly reproduced the recorded forecast-at-100 output. All 20 project tests pass.

The adapter is **not ready for the slider**: identical wording across all five levels increased from 7/30 cases before training to 27/30 afterward. Exact input copies rose from 13/150 to 114/150. Outputs reaching the 160-token cap fell from one to zero. These are mechanical diagnostics, not an independent style rating or semantic accuracy estimate. Spot review found better preservation on the quoted-sign and tentative-meeting examples, but unchanged nicknames at zero and lost style variation at high settings. Training learned a copying tendency more readily than the desired level control.

Retain this pilot as a negative control, not a production replacement. Next, prepare contrastive targets for the same content at all five levels and explicit level descriptions; review actual stylistic differences and preservation of meaning. The weak labels, small dataset, narrow adapter targets and brief numeric prompt are plausible contributors, not established causes. Do not simply add epochs based on lower validation loss. Compare any new configuration with this recorded pilot using the same development inputs and a later untouched test.

`archive-pilot-report.json` retains metrics; raw source snapshots, adapter and before/after texts remain local under ignored `work/`. `assess_pilot.py` regenerates mechanical diagnostics only after both evaluation passes complete.
