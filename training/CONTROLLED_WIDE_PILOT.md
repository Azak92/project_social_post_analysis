# Controlled wider adapter experiment

The working archive-based and prompt-only slider modes remain available with the existing 9B inference model. This is a separate local 2B experiment.

The previous adapter covered q/v projections in six full-attention layers. This experiment covers all 186 `torch.nn.Linear` modules inside the 24 language decoder layers, including linear attention and feed-forward projections. Vision layers, embeddings and output head remain excluded. Rank 8, alpha 16, BF16, seed 42, learning rate 0.0001 and answer-only supervision remain unchanged. The official pinned base is reloaded from disk; earlier experimental adapters are not stacked.

Eight existing reviewed groups give 80 balanced input/target records (plain and styled views at five levels). Eight fixed epochs give 80 optimizer steps. This intentionally small learning-capacity check is not a production fine-tune. Sixty validation records from separate source groups remain for loss measurement; they do not select the checkpoint. The final checkpoint is fixed in advance.

A further editorial pass corrected two targets before optimization: payment remains **for repairing** rather than **to repair**, and the rejected proposal stays explicitly named rather than becoming an ambiguous **it**. These v3 overrides and their hashes are recorded separately; the v2 dataset and run snapshots remain historical evidence. An initial Windows encoding failure and an interrupted pre-optimization baseline are preserved in separate local folders.

Before and after training, eight familiar plain inputs and eight fresh transfer probes are generated at every level: 160 outputs. Familiar inputs test whether the adapter can learn the examples; they must not be reported as evidence of generalization. Transfer probes test payment roles, quantities, uncertainty, negation, attribution, quotation boundaries, critical political language and positive political language. Style and meaning are reviewed separately. This run changes training size and duration as well as adapter scope; it cannot establish that breadth alone caused any improvement.

Run `work/train-2b-env/Scripts/python.exe training/controlled_wide_pilot.py` after unloading Ollama. Existing result directories are never overwritten. The runner records exact code, targets, configuration, data and messages alongside the adapter and generated outputs under `work/training/controlled-wide-v3`. `python training/assess_controlled.py` reports mechanical metrics separately for familiar and transfer cases. All archive use remains within the user's existing local non-commercial authorization.

## Completed result

The 8,409,600-parameter adapter completed 80 optimizer steps in a run lasting 646.67 seconds including evaluation. Peak PyTorch reserved memory during training was 6,637,486,080 bytes; this excludes other processes and display memory. Saved adapter reload reproduced the fixed transfer-payment output. Twenty-seven tests passed.

Familiar-content exact target matches rose from 5/40 to 31/40. All 16 inputs produced some variation across the levels after training, versus 14 before. Five length-limited baseline outputs became zero. These are learning/responsiveness diagnostics, not style or meaning accuracy scores.

Validation loss increased from 0.97505 to 1.23607 (26.77%) while final training loss was about 0.048, consistent with overfitting the deliberately tiny set. The assistant reviewed all 80 after outputs: one clear omission among 40 familiar outputs; ten clear meaning/stance failures and eight minor or ambiguous shifts among 40 transfer outputs. The remaining outputs had no material issue found in this unblinded review, which is not an independent accuracy estimate. Examples include confusing moving with selling, converting reported completion into personal action, losing quoted-text boundaries, and adding a hostile nickname to praise.

This configuration establishes small-set learning capacity, not reliable meaning-preserving generalization. It is not deployed. Both existing 9B slider modes remain unchanged. `/training?run=v3` shows all 160 comparison outputs and per-output review notes; `/five-levels?run=v3` shows the corrected familiar targets. `training/controlled-wide-report.json` records the measured results.

The next useful experiment should retain broad coverage while increasing reviewed content diversity, adding contrastive examples for negation, attribution, quotation boundaries, temporal order and political praise. Use fewer passes with predeclared checkpoint selection based on a separate meaning-and-style development set, followed by fresh unseen probes. Do not optimize against the probes that exposed these errors. An inference-time rejection/retry check can reduce risk but is not a guarantee of semantic equivalence. A matched narrow-versus-wide run would be needed to isolate the causal contribution of adapter scope.
