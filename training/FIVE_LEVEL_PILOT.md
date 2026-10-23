# Five-level pilot v2

This local, non-commercial experiment uses the archive under the user's existing authorization. It does not grant redistribution rights or verify the truth or personal authorship of archive claims.

## What changes

The first adapter mostly copied its input. This experiment supplies all five requested levels for the same content, with both plain and styled input views. Each view has five possible targets, so an identity rewrite only satisfies one target level. Level-specific instructions define vocabulary and cadence, rather than treating the number as a synonym for excitement.

The reviewed local file `data/style/five-level-v2/groups.json` contains 20 archive groups and 10 original everyday examples: 150 targets, yielding 300 input/target records. Of these, 24 groups / 240 records train the model; six groups / 60 records validate it. Archive source-group splits are preserved. Original examples cover uncertainty, negation, roles, attribution and incomplete work. Approval hashes prevent unreviewed edits from silently entering a run. Intermediate targets and original examples are authored and reviewed by the assistant, not independently rated by humans.

Review prioritizes entities, quantities, scope, time, attribution, certainty and stance. It corrected a prediction changed into an established judgment, an invented timing restriction, and ambiguous attribution of a policy position. Some neighboring levels differ only subtly. These are editorial categories, not calibrated percentages of authenticity.

## Fixed experiment

Run `work/train-2b-env/Scripts/python.exe training/five_level_pilot.py` after unloading Ollama. The script refuses to overwrite an existing run. It starts from the pinned official Qwen3.5-2B base, not the failed first adapter. Two fixed epochs, 60 optimizer steps, rank-8 q/v LoRA, BF16, learning rate 0.0001, accumulation eight, and 512-token maximum. Only answer tokens are supervised; overlong examples fail instead of being silently truncated.

Thirty existing development cases are generated at all five levels before and after training with identical instructions and deterministic decoding. They are not training targets. They have been inspected during prior development, so they are not an unseen final benchmark. The original archive test partition remains reserved. A future acceptance test needs fresh content and independent ratings.

The output folder preserves the dataset, messages, exact runner, loader, configuration, adapter, losses and all 300 evaluation outputs. Loss measures target prediction; textual differences measure responsiveness. Neither proves semantic fidelity or characteristic style. Inspect semantic errors, gratuitous repetition, nickname handling and meaning-preserving changes independently. The live slider continues using its existing model until an adapter earns deployment through evaluation.


## Second, five-level pilot (current result)

Reviewed 30 content groups at all five levels: 150 targets, with plain and styled input views producing 240 training and 60 validation records. `/five-levels` displays the reviewed examples; `/training?run=v2` displays all 300 controlled before/after outputs and meaning-review findings. The original experiment remains at `/training?run=v1`.

The fresh 2B rank-8 adapter completed two fixed epochs / 60 optimizer steps. Validation loss fell from 0.9751 to 0.7748 (20.54%). Nevertheless, identical wording across all five levels increased from 5/30 to 16/30 cases; exact input copies rose from 73/150 to 109/150. Length-limited outputs fell from seven to zero. These metrics diagnose behavior, not semantic accuracy.

The assistant reviewed all 150 after-training outputs: five clear omissions/role/attribution errors and three ambiguous completion-to-intent shifts. Low-level nickname removal works on one case, but high-level style transfer is generally weak. The adapter was saved and reload-verified, but is not deployed. The live slider remains the existing 9B model. Twenty-five tests pass. Details and limitations are in `training/FIVE_LEVEL_PILOT.md` and `training/five-level-pilot-report.json`; run artifacts remain local under `work/training/five-level-pilot-v2`.

Next experiment: first test whether a wider adapter can deliberately learn a small, balanced five-level fixture. The current adapter touches only q/v projections in six full-attention layers. An expanded set of language layers is a hypothesis to test, not an established cure. Check memory and training-fit style control before spending another run on more data. Then add more varied, independently reviewed examples and new unseen evaluation cases. Do not judge success from loss or copying.
