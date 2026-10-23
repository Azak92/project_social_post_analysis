# Paused session — 23 October 2026

## Resume the site

On this computer, double-click `C:\IT\project_social_post_analysis\Open Language Lab.cmd`. It starts or reuses the local services and opens http://127.0.0.1:8000/. If services are already running, opening that address directly also works. After a restart, use the launcher first. No training starts automatically.

The first rewrite loads the existing 9B Ollama model and may take longer. Merely browsing saved results does not load it. The homepage's Release GPU memory button unloads the inference model; generations otherwise request a five-minute idle keep-alive. The website and idle runtime can remain running without holding model weights on the GPU.

## Saved state

- Both archive-example and prompt-only slider modes remain on the existing 9B model, with levels 0/25/50/75/100.
- The separate Qwen3.5-2B training environment, pinned base model, all adapters and run snapshots are preserved locally in `work/`.
- Archive data, reviewed examples and evaluation exports remain locally in `data/`. These ignored folders are **not backed up by the GitHub push**. A fresh clone needs these local assets or a separate setup/import process.
- Source code, tests, launchers, dependency locks, source-use notes and numeric experiment reports are committed to GitHub. Existing untracked notebooks and editor-generated files were not part of this implementation commit.

## Latest experiment

`work/training/controlled-wide-v3` holds the broader-adapter test: 186 projections across 24 language layers, 8,409,600 trainable parameters, eight reviewed groups, 80 training records, eight fixed epochs / 80 optimizer steps. The adapter is experimental and not deployed.

Familiar target matches improved from 5/40 to 31/40, but validation loss rose from 0.97505 to 1.23607. Review found one clear omission in 40 familiar outputs; ten clear meaning/stance failures and eight minor or ambiguous shifts in 40 transfer outputs. Saved adapter reload passed, and 27 tests passed. Details: `training/CONTROLLED_WIDE_PILOT.md` and `training/controlled-wide-report.json`.

Review pages: `/training?run=v3` and `/five-levels?run=v3`. Earlier runs remain available using `run=v1` and `run=v2` on the training page.

## Next discussion

Keep the broader adapter, increase reviewed content diversity, add contrasts for negation, reported claims, quotation boundaries, payment timing and positive political statements, and reduce repeated passes over the same examples. Predeclare checkpoint selection on separate meaning-and-style development examples and use new transfer probes afterward. Do not claim reliability from loss or memorization. The user has authorized local non-commercial archive training; no renewed approval is needed for that existing scope.

Paused at the user's request. No further training or background follow-up is scheduled. At shutdown checks, Ollama reported no loaded models and no training/verification Python process remained.
