# Archive-use check — 23 October 2026

## User scope decision

After reviewing these findings, the user explicitly authorized use of the archive for this local non-commercial training project. The one-epoch pilot proceeds within that scope. This records the user’s decision; it does not convert the archive into an openly licensed dataset or authorize publication of the corpus or model. The original findings below remain a record of what the checked documents establish.

## Finding

No explicit license granting training rights to the underlying Truth Social posts was established by the documents checked. This is a bounded terms review, not a legal determination that local research is prohibited or permitted. The environment test can proceed on original fixtures without resolving this question.

## Evidence

1. The installed Hugging Face archive, revision `dde19cab0f3c5d783b9bc863700ecdd5a47d605c`, is marked `license: other`. Its retained NOTICE says the publisher does not claim rights to relicense underlying text/media and asks downstream users to verify upstream terms before training. A dataset download and checksum do not supply a license from the post rights holders.
2. Trump’s Truth About and FAQ pages describe research/journalistic access and an RSS feed. Neither of those pages provides an explicit grant for model training. Their research purpose alone should not be interpreted as such a grant. We did not contact the operator or accept new terms.
3. Truth Social's current terms, section 2, describe a limited personal non-commercial access/download license; sections 4 and 7 restrict automated access and systematic collection. Section 9 grants broad contribution rights to the platform. That platform license is not automatically a downstream training license for this project. How those provisions and any statutory exceptions apply to an independently obtained archive requires a separate legal assessment. The app has not begun direct platform scraping.
4. The official Qwen/Qwen3.5-2B model card declares Apache-2.0. That covers the model under its license terms, not the archive used as future training material. Retain the downloaded model license.
5. This review covers the selected Truth Social source. It does not establish rights for the older Twitter CSVs or their underlying posts; do not silently switch to them as a supposedly cleared source.

## Original readiness recommendation (superseded for the authorized local pilot)

Prepare and verify the 2B environment with original, non-archive fixtures. Keep all existing archive review records and provenance intact. Before an archive-based fine-tune, resolve a documented basis for the intended use: appropriate rights-holder permission, a suitable license, or an informed assessment of applicable legal exceptions. An archive operator may not own the underlying post rights. No requests for permission have been sent on the user's behalf. Do not label reviewed pairs as legally cleared simply because their meaning was checked.

Sources checked:

- https://huggingface.co/datasets/Cameronk199/donald-trump-truth-social-posts
- Local pinned `data/raw/truth/dde19cab0f3c5d783b9bc863700ecdd5a47d605c/NOTICE.md`
- https://www.trumpstruth.org/about
- https://www.trumpstruth.org/faq
- https://help.truthsocial.com/legal/terms-of-service/ (sections 2, 4, 7, 9)
- https://huggingface.co/Qwen/Qwen3.5-2B
