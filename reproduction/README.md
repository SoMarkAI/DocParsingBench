# Reproducing a recorded run

This directory supplies the prediction-side adapters used in recorded runs and
checks that the scoring inputs are complete and byte-identical. It does **not**
change DPB segmentation, matching, metrics, weights, aggregation, chemical-page
exclusions, or formula fallback. Nothing here runs automatically in `dpb eval`.

Prediction replay and fresh inference are different. Replay reconstructs Markdown
from saved outputs and can be exact. Fresh inference also needs the same model,
layout weights, templates, inference code and runtime. A model name alone is not
enough. `historical-inputs.json` records the audited September–October 2026 inputs;
it is not a new public leaderboard and does not replace the older README rows.

## Preparation (no ground-truth access)

Run from this repository checkout. Preparation and verification use only the
Python standard library and need no GPU.

| Profile | Input | Scope |
| --- | --- | --- |
| `identity-v1` | Flat directory of Markdown files | Byte-identical copies, including repetitions and non-empty truncated output. |
| `display-math-v1` | Flat directory of Markdown files | Historical OXR adapter: blank-line boundaries around existing complete `$$…$$` / `\[…\]` outside matched HTML tables. Formula payloads, matched tables and non-whitespace characters are checked for preservation. |
| `surya-html-v1` | Saved Surya page JSON with a `blocks` list | Historical serializer: reading-order blocks, HTML math to LaTeX delimiters, `h1`–`h6` to Markdown headings, `SectionHeader` to level 3 when no source heading level exists. |

Surya's historical policy omits blocks flagged `skipped` or `error` (the count is
recorded), HTML-unescapes math/heading payloads, and preserves table blocks starting
with `<table`. Malformed nested HTML and table-internal math are not comprehensively
repaired. This is not a general HTML converter.

`display-math-v1` retains the historical regex behavior, including the lack of
code-fence protection. Do not enable it globally or apply it to every model.
Changing these rules requires a new profile and re-evaluation, not silently
replacing historical inputs. Already isolated formulas are unchanged.
Like the historical runner, the display-math profile reads text in Python's
universal-newline mode (CRLF becomes LF); identity preserves raw newline bytes.

```bash
# OXR: source is the original Markdown, not already adapted predictions.
python -m reproduction.prepare \
  --source /path/to/original-oxr-markdown --output /path/to/new-oxr-predictions \
  --profile display-math-v1 --expected-count 1400

# Surya: source is the saved page JSON, not the old Markdown export.
python -m reproduction.prepare \
  --source /path/to/surya-responses --output /path/to/new-surya-predictions \
  --profile surya-html-v1 --expected-count 1400

# Other official Markdown exports need no external adapter.
python -m reproduction.prepare \
  --source /path/to/official-export --output /path/to/new-predictions \
  --profile identity-v1 --expected-count 1400
```

Output must not already exist or overlap the source. Files are validated in a
temporary sibling directory before publishing the complete directory; failure
leaves no partially prepared output. `preparation.json` records implementation
hashes, per-file/collection hashes, serializer counts and changed Markdown names.
Surya JSON-to-Markdown conversion is not counted as "changed Markdown files".

## Verify the inputs, then score

Use the matching model's `prediction_collection_sha256` from the manifest. The
October 1 runtime-corrected Paddle predictions supersede the September 30 set.

```bash
python -m reproduction.verify \
  --gt /path/to/frozen-dataset/markdowns --pred /path/to/prepared-predictions \
  --config reproduction/scoring.yaml --expected-count 1400 \
  --expected-gt-sha256 25185917d61f2b0f86dd0698e49f61b559a16cccaf3e8584db4826345050881f \
  --expected-pred-sha256 MODEL_PREDICTION_COLLECTION_SHA256 \
  --expected-config-sha256 361a55280d032c4e534dab336b309b17c2a750c837936be7d03fe28c6a4a53b2
```

Missing, empty or whitespace-only pages, differing filename sets, counts or hashes
fail verification. It reports `scoring_verification: not_run`; this is not a score.
Collection hashing sorts flat filenames and appends each UTF-8 filename, NUL,
the **binary** SHA-256 digest of raw file bytes, and newline to one SHA-256 stream.
It is not the hash of a text `SHA256SUMS` manifest.

The frozen HF dataset revision is `907ca67e0dce06cf2bfcc51b3335b0b3b1f3e3f7`.
Original DPB receives 1400 predictions; 58 GT pages containing `\smiles{` are
excluded by its existing policy, leaving 1342 eligible reports.

The scorer is DPB `63d80f96a89e2f2514daac703238d6f12a2873e9` plus FastCDM
`ef5206caa4de87abbf39510678d12681c20a74f2`. This change leaves `docparsingbench/`
identical to that DPB revision. In an isolated environment:

```bash
python -m pip install .
python -m pip install 'fastcdm @ git+https://github.com/SoMarkAI/FastCDM.git@ef5206caa4de87abbf39510678d12681c20a74f2'
# Check FASTCDM_SRC does not override the pinned source unexpectedly.
python -c 'import os; print("FASTCDM_SRC:", os.environ.get("FASTCDM_SRC"))'
python -m docparsingbench.cli eval \
  --gt /path/to/frozen-dataset/markdowns --pred /path/to/prepared-predictions \
  --config reproduction/scoring.yaml --out /path/to/new-result.json
```

Install compatible Chrome/ChromeDriver first, following the repository's setup
instructions. The recorded driver path is `/usr/bin/chromedriver`. If yours differs,
copy the YAML, change only that path, record the diff and both hashes, and verify
using the actual config hash. The base YAML is retained byte-for-byte, including
chart options; plotting does not define the score.

Check browser/driver versions, writable runtime directories and formula rendering
before a full evaluation. Browser crashes are not a model score. Keep the original
fallback policy and record render failures. Retain full code commits, `pip freeze`,
image identity, browser versions, prediction/config/result hashes and report counts.
Do not average the leaderboard category columns to reconstruct DPB: its existing
aggregation uses eligible per-page reports and category-bearing pages separately.

FastCDM at this revision calls RANSAC without a fixed RNG. Identical predictions can
produce small formula/overall variations. This PR does not patch the scorer. Keep
the original result JSON/hash as the record of a particular run; report observed
rerun variation, not an unmeasured tolerance or a selected favorable run.

## Inference omissions found in the audit

- **OXR-1.0:** historical layout weights must match, not only the OXR checkpoint.
  Newer public layout weights are a different experiment. The included reading-order
  patch removes duplicate IDs only from otherwise complete permutations, preserving
  first occurrence; it is not Markdown deduplication.
- **TeleOCR:** use its pinned official two-step parser and OTSL converter. The
  included patch initializes `lang = DEFAULT_LANG`; three failed pages were rerun.
- **PaddleOCR-VL-1.6:** use its v1.6 pipeline, model-supplied chat template,
  PP-DocLayoutV3 and recorded processor settings. The formula fix was in inference,
  not a Markdown adapter. Export is official `save_to_markdown`, outer whitespace
  stripping plus final newline; external preparation is identity.
- **MinerU:** recorded vLLM `0.14.1.dev1`, Transformers `4.57.6`,
  `mineru-vl-utils` `1.0.5`; no external formula adapter was needed.
- **dots.mocr / Infinity-Parser2-Pro:** keep their pinned structured-output
  converters, including dots' non-empty raw fallback and Infinity's incomplete
  element/header/footer policies. Generic JSON-to-Markdown is not equivalent.
- **OvisOCR2 / Qianfan-OCR:** keep non-empty `finish_reason=length` outputs,
  including repetitions or truncation. Do not remove repeated text. Their original
  runner also stripped outer whitespace; Ovis removed blocks starting with its
  image placeholder prefix. That can remove accompanying text in the same block,
  so the exact rule is recorded rather than described as lossless raw output.
  Their historical `responses` files contain status/usage/checksum metadata, not
  complete API responses. Raw-response replay cannot be checked from those files.

The manifest records weights/templates, source commits and effective settings.
Compatibility patches are under `patches/`; apply them only to their recorded
source revisions (`git apply --check` first), not a newer tree blindly.

## What is verified, and what still needs publishing

Local replay rebuilt all 1400 retained OXR and Surya predictions byte-for-byte.
Identity copying matched all 1400 MinerU and runtime-corrected Paddle predictions.
All nine audited inputs have matching page names and frozen fingerprints; retained
aggregates were checked against original DPB aggregation. This was a replay/input
audit, **not fresh inference or a new full Chrome/FastCDM scoring run**.

This PR includes no raw responses, prediction bundles, private checkpoints or
private leaderboard scores. The public repository alone is therefore **not yet a
complete end-to-end historical reproduction package**. Before claiming a public
row is independently reproducible, publish or provide:

- immutable predictions/raw-response artifacts and hashes, or a fully available
  fresh-inference recipe;
- exact accessible model/layout/template assets, including historical OXR layout
  weights if reproducing that run;
- the inference runner and effective prompts/converter settings;
- a downloadable immutable evaluator image or full environment lock with browser
  versions (a local image ID is not a download link);
- retained result JSON/hash and any rerun variation.

Missing artifacts must stay marked as missing. Their release needs a separate
publication decision; local replay alone does not prove public reproducibility.

## Tests

```bash
python -m unittest discover -s tests -p test_reproduction.py -v
# With the repository's test dependencies installed:
python -m pytest -q
```
