# Trump Language Lab

A local-first research playground for public-post language and synthetic style transformation.

Current experiment: the broader 2B adapter has completed a controlled eight-group test. It learned 31/40 familiar targets exactly, but separate transfer review found meaning and stance errors, so it remains experimental. Both existing 9B slider modes are unchanged. View `/training?run=v3` for results and `/five-levels?run=v3` for corrected targets; details are in `training/CONTROLLED_WIDE_PILOT.md`. Earlier sections below retain the progression of prior experiments.

## Run locally

The rewrite slider has separate prompt descriptions for 0, 25, 50, 75 and 100: plain wording, subtle conversational cadence, recognizable restrained phrasing, strong rhetorical structure, and full Trump-inspired delivery. Both generation modes use the same level descriptions. Archive mode adds retrieved examples as style evidence. Higher levels prioritize vocabulary and clause rhythm over capitals; nicknames remain optional and restricted to existing political criticism. These are prompt controls, not trained or empirically calibrated style scores, and meaning preservation still needs output review.

```powershell
python -m pip install -r requirements-app.txt
python -m lab.data --truth
python -m uvicorn lab.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Twitter CSVs must be present in `data/` (already supplied locally). Raw data and the SQLite database remain ignored by Git. The app uses FastAPI, SQLite, and plain HTML/CSS/JavaScript. It makes no external requests during ordinary analysis or browsing. Rewriting connects only to Ollama on localhost, and is unavailable until a model is installed. No cloud fallback or fake generated output.

## Sources and updates

Twitter: `realDonaldTrump_bf_office.csv` and `realDonaldTrump_in_office.csv`. The `ID` column is an account handle; actual post IDs are extracted from source URLs and stored as strings. Source times have no declared timezone and remain explicitly unknown. `RT @` is only a heuristic repost flag; other Twitter rows have unknown type/authorship.

Truth Social: https://huggingface.co/datasets/Cameronk199/donald-trump-truth-social-posts

Default revision: `dde19cab0f3c5d783b9bc863700ecdd5a47d605c`. Imports download the text archive, dataset card, notice, and manifest; verify publisher-provided SHA-256 checksums; retain immutable revision folders and retrieval receipts. Checksums establish consistency with the publisher, not independent truth or authorship. The dataset uses `license: other`; review its retained NOTICE before redistribution or training. No media binaries are downloaded.

```powershell
python -m lab.data --truth --revision latest
```

This resolves and records the current commit, downloads its snapshot, and upserts by platform/post ID. It supports new and corrected rows without duplicate records. It is a snapshot reconciliation hook, not a live scraper or automatic schedule. Missing rows are retained, not treated as deletions. Source adapters can later add direct incremental retrieval with overlap windows and checkpointing. The archive's coverage may lag its publication date.

## Learning and evaluation roadmap

1. **Data hygiene (implemented baseline):** preserve original text, repair encoding separately with ftfy, record provenance, distinguish reposts, and preserve unknown timezones. Check missing text, repeated text, date ranges and source overlap before modeling.
2. **Style features (implemented baseline):** capitalization, punctuation, word length and vocabulary share. These are interpretable descriptions, not sophistication or intelligence ratings. Short-post readability and vocabulary measures are unstable; compare matched lengths and report uncertainty.
3. **Meaning (next):** benchmark a small sentence-embedding model for semantic search. Compare against lexical search on a manually judged query set. Examine clusters as hypotheses; do not equate a topic cluster with an author.
4. **Style transformer (local model and archive reference modes implemented):** compare local instruction models on a fixed set of neutral, factual, uncertain and already-emphatic posts at 0/25/50/75/100 intensity. Judge meaning preservation, unwanted new claims, fluency, style strength and latency. Embedding similarity alone will not catch negation or number changes. The slider is a prompt control, not a learned probability.
5. **Authorship (research only):** no verified author labels exist in the supplied CSVs. Device, era and posting time can be confounds, not ground truth. Start with style variation and outliers. Any future supervised experiment needs defensible labels and held-out, time-aware evaluation.
6. **Fine-tuning (decision later):** only after a prompt-only baseline and data-rights review. Build and review paired neutral/stylized examples; raw posts alone do not teach meaning-preserving transformation. Group near-duplicates before splitting, reserve a chronological test set, and compare LoRA/QLoRA against the unchanged base model. Discuss model, quantization, memory, training targets and evaluation before training.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The generation tests mock the model service; they do not establish real model quality. Keep the server bound to loopback; this is a personal local prototype without hosted authentication.

## Local model baseline

The selected baseline is `qwen3.5:9b-q4_K_M` through Ollama. Runtime and weights live in ignored `work/` folders. Start both local services with `./start-lab.ps1`. Runtime configuration binds to loopback, disables cloud features, and allows one parallel generation. The app requests 4096 context tokens, non-thinking output, temperature 0.4 and seed 42. Seed improves repeatability but does not guarantee identical output across runtime/hardware versions.

The 9B model is a starting hypothesis about quality versus speed, not a demonstrated winner over other models. Check actual GPU residency and latency on this machine. A 4B model is the fallback for lower latency or less free VRAM. The model is not trained on the imported archive: this version uses instruction-based style transfer. Sliding requests style strength (0 ordinary wording to 100 full archive-inspired style); it is not a calibrated Trump-likeness score. Transform once, then release the slider to regenerate from the original input.

First evaluate preservation of numbers, names, negation and uncertainty, then style strength and fluency. A stronger rhetorical tone may still distort emphasis even when words and numbers survive. Automated unit tests check the integration; human comparison remains necessary for meaning preservation.

Final setup verification (23 October 2026): Ollama 0.40.9; model digest 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90. Ollama reported 5,606,337,412 bytes allocated entirely on GPU at 4096 context. Nine final short rewrites completed in 0.41–0.82 seconds after warmup. Cold start previously took 49 seconds. These are smoke measurements, not a broad benchmark. Reasoning mode exhausted the 2048-token trial budget before producing a usable rewrite, so fast mode remains selected. Basic number, uncertainty-word and negation review flags are shown, but do not certify semantic fidelity. High intensity can still invent emphasis or details; manual review remains necessary. See the latest validation below for the current test count.

## Archive-conditioned style and training-pair pilot

The slider now defaults to **Real archive examples**; **Compare both** runs archive mode and the old prompt-only baseline on the same input, intensity, model and sampling settings. The displayed source excerpts make retrieval inspectable. These modes change prompts, not weights. The comparison measures the complete archive-assisted prompt (examples plus instructions and a source-backed nickname rule), not the isolated effect of retrieval alone.

Build the source index with `python -m lab.archive_style`. It uses short Truth Social originals, strips trailing signatures, excludes links/handles and endorsement boilerplate, and retains provenance plus the original text. Four-word shingle Jaccard similarity >=0.65 groups near-duplicates before splitting by each group's newest timestamp: training before 1 June 2026; validation in June; test from 1 July. Grouping is heuristic and does not catch every paraphrase. Current index: 6,662 posts / 6,603 groups; 6,375 train, 152 validation, 135 test. Retrieval uses only train, further limits examples to 60 words, ranks lexical overlap and rhetorical category, and avoids examples categorized as political for nonpolitical input. Topic detection can still miss political references. No embedding model is used yet.

At high intensity, criticism of Democrats can use the source-attested spelling “Dumocrats” when retrieved evidence contains it. This is an explicit vocabulary rule; it is not learned behavior. At low intensity, the model is asked to remove distinctive nicknames and rhetorical patterns while preserving stance. Intensity remains a requested level, not a measured style probability. The model can still introduce claims or exaggerations: inspect both outputs.

Generate or resume the pilot with `python -m lab.prepare_pairs`. It selects 60 diverse source groups (40 train, 10 validation, 10 test), generates neutral and medium versions locally, and writes individual provenance-rich records plus `train.draft.jsonl`, `validation.draft.jsonl`, and `test.draft.jsonl` under ignored `data/style/pairs/`. Each pair produces three draft messages: original→neutral at 0; neutral→synthetic medium at 50; neutral→original at 100. The original's 100 label is a provisional proxy, not a measured label; some originals are relatively mild. Intermediate labels are synthetic and require calibration.

The expanded pilot contains 120 candidate pairs, all reviewed by Codex against the supplied source text: 75 accepted after corrections and 45 excluded. Accepted pairs yield 225 message examples: 168 train (56 sources), 30 validation (10), and 27 test (9). These counts are a local snapshot; `data/style/reviewed/manifest.json` is authoritative. Review does not independently authenticate the posts, verify their factual claims or establish content rights. These are small, topic-biased chronological holdouts, not a representative benchmark.

Important corrections include entity substitutions (A.G. James incorrectly became Pam Bondi), certainty changes, accusations assigned to the wrong speaker, missing criticism, and planned work incorrectly described as completed. Exclusions include fragments, ambiguous quotations, unclear references, and examples with too little style contrast. Neutral means less characteristic delivery with the same stance; it does not mean agreeable, politically balanced or fact-checked. Medium targets are editorial paraphrases, not independent evidence that intensity 50 is calibrated.

Local decisions are retained in `data/style/reviews/pilot-decisions.json` and `expanded-decisions.json`; before-edit records remain in `data/style/pairs/history/`. The review page at `/pairs` shows decisions and corrections and filters accepted, excluded and pending pairs. Re-running generation preserves existing records and their reviews.

`python -m lab.prepare_pairs --expand` adds up to 60 previously unused source groups (48 train, 6 validation, 6 test), using shorter sources and excluding obvious quotation/appointment patterns. This is a quality-focused batch size, not a target for final training scale. Newly generated pairs remain unreviewed and cannot enter training exports. Run `python -m lab.review_pairs --decisions data/style/reviews/expanded-decisions.json` to apply that specific saved review, or `python -m lab.review_pairs` to refresh exports. Use new decision files for future batches rather than reapplying old edits.

Only `data/style/reviewed/{train,validation,test}.jsonl` contains approved examples. Each exported record carries source ID, group, URL, intensity and a SHA-256 fingerprint of the reviewed source and targets. Any later content edit invalidates approval until reviewed again. Cross-split source groups cause export to fail. Draft JSONL files remain inspection material and must never go directly into training. The three transformations of one pair are correlated; 225 records are only 75 independent source pairs. Holdout target corrections do not train weights, but this small curated test set will need a larger fresh test set before strong quality claims. No fine-tuning has run.

Validation: ten unit/integration tests; twelve live comparisons covering politics, team praise and uncertain deadlines at both slider endpoints. Archive examples changed vocabulary and cadence, but some outputs still added unsupported praise or rhetoric. This is a demonstrable style baseline, not a claim of proven semantic fidelity or broad quality improvement.


## Everyday startup and GPU memory

Double-click `Open Language Lab.cmd` in this repository to start the services and open http://127.0.0.1:8000/. The shortcut uses the installed Python and local Ollama files, waits for both services, and reuses services already running. It does not configure automatic startup at boot. After restarting or shutting down the PC, use the shortcut again; a browser URL alone cannot start a stopped server. On this PC, closing the browser or Codex does not intentionally stop the separately launched services. Sleeping the PC makes the site unavailable until it wakes. The loopback address works only on this PC.

Every generation explicitly requests a five-minute idle keep-alive. While loaded, the model occupies GPU memory even between requests, although residency does not mean active inference. After the idle timeout it unloads from GPU memory; model files remain on disk. The homepage shows current residency and includes **Check memory** and **Release GPU memory** controls. Reading the memory status or browsing posts does not load the model. Releasing memory leaves the website running; the next rewrite automatically reloads the model and may take longer. The model runtime remains a background process with some normal system RAM use even when no model is loaded.

Latest validation: 14 unit/integration tests pass, including reviewed-only export, stale-approval rejection, cross-split leakage rejection and explicit keep-alive/unload behavior. Dataset exports contain 168 / 30 / 27 examples with disjoint source groups. UI filtering and live model load/release are checked separately; these checks do not measure trained-model quality.

Live verification: the browser rewrite loaded the model (5,606,337,412 bytes on GPU), completed in 5.07 seconds, and the Release GPU memory button returned an empty loaded-model list while the site and output remained available. The startup script also successfully reused both running services.


## Five-position evaluation and training readiness

The slider and API now accept only 0, 25, 50, 75 and 100. `/evaluation` displays a fixed 30-case development benchmark, with 300 saved outputs across five levels and two modes. All results were generated locally with unchanged model weights. The runner, original cases and meaning/style rubric are in `lab/evaluate.py`, `lab/evaluation_cases.py` and `TRAINING_PLAN.md`. Saved results, runtime/model metadata, implementation snapshot, retrieval-index hash and diagnostic review notes remain under ignored `data/evaluation/baseline-v1/`.

Automatic flags appeared in 29/150 baseline outputs and 36/150 archive outputs; these are neither error rates nor accuracy scores. Spot review found missed errors: extra motives and consequences at 100, unchanged nicknames at 0, and an inspection changed from completed to pending. Positive cases also exist; do not infer that every output fails. Before tuning, independently score meaning, fluency and observed style, and evaluate all five levels. Current reviewed training labels cover only 0/50/100.

Training-readiness findings: RTX 4070 SUPER with 12 GB VRAM; existing Transformers is v4, while current Qwen3.5 training guidance requires v5. The proposed local experiment is 4B BF16 LoRA subject to measured memory fit, with 2B as fallback. Unsloth discourages Qwen3.5 QLoRA and estimates 22 GB for 9B BF16 LoRA. Source-use terms remain unresolved. No training dependencies or additional model weights have been installed. See `TRAINING_PLAN.md` for the proposed smoke test, environment isolation and evaluation gates. Sixteen tests pass after the five-position change.


## Prepared 2B training environment

The isolated environment under `work/train-2b-env` now passes CUDA/BF16, dependency and model-import checks. Official Qwen3.5-2B weights are pinned locally; setup instructions, package lock, source-use review and measured results live in `training/`. A ten-step LoRA test on newly written non-archive fixtures passed, as did adapter reload and a 512-token synthetic backward pass. Eighteen tests pass. This validates the local training machinery, not learned Trump style. Archive-use permission has not been established by the checked documents; see `training/ARCHIVE_USE_REVIEW.md`. The website's dependencies and 9B inference model are unchanged.


## Archive pilot result

Following explicit user authorization for local non-commercial archive training, one 2B BF16 LoRA epoch completed on the 168 reviewed training messages. `/training` displays 300 matched before/after outputs. Validation loss improved 12.55%, but level control worsened: 27/30 cases produced identical output at all five levels, compared with 7 before training. The adapter is saved as an experiment and is not deployed to the live slider. See `training/README.md` and `training/archive-pilot-report.json` for measurements and next-experiment rationale. The earlier readiness sections describe the pre-training state; this section is the current status. Twenty tests pass, and the saved adapter reload was verified.


## Second, five-level pilot (current result)

Reviewed 30 content groups at all five levels: 150 targets, with plain and styled input views producing 240 training and 60 validation records. `/five-levels` displays the reviewed examples; `/training?run=v2` displays all 300 controlled before/after outputs and meaning-review findings. The original experiment remains at `/training?run=v1`.

The fresh 2B rank-8 adapter completed two fixed epochs / 60 optimizer steps. Validation loss fell from 0.9751 to 0.7748 (20.54%). Nevertheless, identical wording across all five levels increased from 5/30 to 16/30 cases; exact input copies rose from 73/150 to 109/150. Length-limited outputs fell from seven to zero. These metrics diagnose behavior, not semantic accuracy.

The assistant reviewed all 150 after-training outputs: five clear omissions/role/attribution errors and three ambiguous completion-to-intent shifts. Low-level nickname removal works on one case, but high-level style transfer is generally weak. The adapter was saved and reload-verified, but is not deployed. The live slider remains the existing 9B model. Twenty-five tests pass. Details and limitations are in `training/FIVE_LEVEL_PILOT.md` and `training/five-level-pilot-report.json`; run artifacts remain local under `work/training/five-level-pilot-v2`.

Next experiment: first test whether a wider adapter can deliberately learn a small, balanced five-level fixture. The current adapter touches only q/v projections in six full-attention layers. An expanded set of language layers is a hypothesis to test, not an established cure. Check memory and training-fit style control before spending another run on more data. Then add more varied, independently reviewed examples and new unseen evaluation cases. Do not judge success from loss or copying.

## Topic generator (25 October 2026)

The homepage now includes a topic-to-post generator alongside the two-method rewrite tool. `/api/topic` takes `topic` (1–300 characters) and an installed local `model`. It always requests full Trump-inspired style and one original post no longer than 280 characters.

Retrieval uses the existing train-only curated archive index, with lexical overlap, broad category and style scores. For topic generation, political archive entries are eligible for a neutral-category query only when they share a query word; existing rewrite retrieval behavior is unchanged. The top match and up to two randomly chosen other top-eight matches supply style context. There is no embedding index or live news lookup.

The existing 9B LLM generates with temperature 0.75 and a fresh random seed, without a fine-tuned adapter. References are style data, not factual grounding. Prompts request general opinions/wishes and prohibit invented current-event details, but this is not a factual verification mechanism. The visible output is labelled generated parody, not a real Trump post or verified news. Repeated clicks can produce different takes; uniqueness is not guaranteed.

Empty, overlong, truncated or substantially verbatim copied outputs trigger one fresh-generation retry, then a visible error if still invalid. Two independent generation flows share a frontend busy state to avoid overlapping GPU requests. The normal rewrite tool still generates both methods. Thirty-two tests pass, including topic validation, retry/copy behavior, archive context and political retrieval isolation. Live tests covered gas prices, Iran and coffee, plus two different outputs for the same topic.
