---
title: "WGO-Bench ego demo on Windows"
description: "Run a small first-person WGO-Bench clip using project-local configuration, Python, and outputs"
---

Use this local demo to inspect the contact sheets and run Refiner's two-pass
subtask annotation on the first 26.54 seconds of WGO-Bench's `homer_1` episode.
The dataset's five gold segments are displayed for comparison and are never
passed to the model. This cropped demo is not a full benchmark evaluation.

## Layout

All task files live under `semantic annotation`:

```text
semantic annotation/
  datasets/wgo-bench-ego/
    clip.mp4                 # 26.54-second demo clip
    homer_1_source.mp4        # One downloaded source episode
    sample.json              # Provenance and gold segments
    .cache/wgo-ranges/        # Downloaded Parquet byte ranges
  refiner/
    .env                     # Local Gemini API key
    .env.example             # Blank configuration template
    run_wgo_demo.ps1          # Windows launcher
    examples/wgo_ego/run_demo.py
    .runtime/python/         # Project-local Python runtime
    .venv/                   # Independent project environment
    artifacts/wgo-ego-demo/   # Preview, contact sheets, predictions and logs
    .work/                   # Temporary files and setup history
```

## Configure and run

For a fresh checkout, install Python 3.12, then create the environment inside
the repository and install the demo dependencies from its root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[video]" python-dotenv
Copy-Item .env.example .env
```

Copy the `.env` template only when `.env` does not already exist. This machine
already has a project-local Python runtime and environment; do not recreate
them just to rerun the demo. A fresh checkout does not include the runtime,
dataset, credentials, or past API captures. Place `clip.mp4` and `sample.json`
in the sibling `datasets/wgo-bench-ego/` directory before running. `sample.json`
must contain `id`, `instruction`, `clip_duration_sec`, and `gold_segments`
(objects with `start_sec`, `end_sec`, and `subtask`). Use the pinned source
and original GT described below; no labels are passed into model inference.

Edit `.env` in the `refiner` project root and fill in:

```dotenv
GOOGLE_GENERATIVE_AI_API_KEY=your_key_here
```

Existing process environment variables take priority over `.env`. The real
`.env`, environment, runtime, outputs and temporary files are ignored by Git.

From a PowerShell terminal in `refiner`, generate the contact sheets without
making model requests:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_wgo_demo.ps1 -Prepare
```

Run both model passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_wgo_demo.ps1
```

Both passes default to `gemini-3.8-flash`. Override it with `-Model` if needed.
The model ID was checked against the
[official model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
on 2026-10-10. Account access is required for actual inference.

Open `artifacts/wgo-ego-demo/preview.html` to view the clip, input contact sheets
and gold segments. After inference succeeds, the same page includes predictions.
`segmentation.json` stores the first pass; `result.json` stores both passes.
Each inference run has a separate folder under `artifacts/wgo-ego-demo/runs/`.

The example uses one worker per stage and one concurrent model request. No local
GPU is required. A fresh environment needs the project's `video` extra and
`python-dotenv`; this machine's project environment is already prepared.

## Record complete API inputs and outputs

Run `.\run_wgo_demo.ps1 -RecordApi` to save every actual HTTP attempt under
the new run's `api/` folder. Open `api/index.html` to inspect all requests in
order, including any failed attempts. API keys are excluded.

Each request folder contains the complete `request.json` (including image
base64), `request.readable.json`, `prompt.txt`, the exact decoded image files,
`generation-config.json`, and `output-schema.json`. The response is stored as
the original `response.raw.json` and formatted `response.json`. Successful
JSON model text is also extracted to `model-output.json`. `metadata.json`
records HTTP status, block reason, model version, response ID, token usage,
image sizes and hashes. `index.json` indexes all requests. A completed run
also stores its parsed two-pass result in its own `result.json`.

The recorder observes the normal worker's HTTP calls without altering the
prompts, image bytes, model, or response parsing. A blocked first pass has no
second-pass requests or successful annotation outputs.

## Internal Notes

The Git repository root is the code project (`refiner` in the local layout).
The original Refiner license and source attribution are retained. `origin` points to
`youliang-zhu/ego-semantic-annotation`, and `upstream` points to
`macrodata-labs/refiner`. Local data, `.env`, environments, caches, scratch work
and API recordings are ignored. The upstream PyPI release workflow runs only
in the upstream repository, preventing this experiment from publishing the
original package on push.

The published repository starts from a clean snapshot of upstream commit
`5f76982` plus this demo. GitHub push protection found an OpenRouter credential
in the inherited upstream history (`4f4f275`, `examples/inference_endpoint.py`),
so that history is excluded from the published `main` branch. The original
local history is backed up in ignored `.work/refiner-before-publication.bundle`.
That backup and the upstream refs are for local reference, not for publication;
push only this repository's `main` branch.

### Original ground truth and evaluation

The sibling dataset folder also contains `homer_1.original.parquet`, a
reserialized copy of the complete first source row, including video. Its
fields and values were verified equal to the pinned source row.
`homer_1.gt.original.json` preserves all non-video fields and all 20 segments;
`clip.gt.original.json` contains the first five segments verbatim, without
renaming fields, translating labels, rounding times, or changing boundaries.
`gt.provenance.json` records the source revision and export details.

The preview displays the original GT JSON and links to these files. The
requested clip ends at 26.514 seconds; encoding produces 26.536077 seconds.
The original GT is not extended to cover that difference.

This repository does not include the blog's benchmark evaluator. The blog
describes Segment F1 at temporal IoU >= 0.75, snapping predicted outer
boundaries to gold boundaries, and a Gemini 3.5 Flash label judge. Its rubric
is published under Grading in the linked blog. The demo's Gemini 3.8 Flash
is the annotation model; it does not automatically become an evaluation judge.

The API run on 2026-10-10 was blocked during segmentation with
`promptFeedback.blockReason=PROHIBITED_CONTENT`, including the built-in retry.
No predictions or evaluation scores were produced. This is a failed inference
run, not a zero benchmark score. The preview and `run-status.json` record this
outcome; the original GT remains available for inspection.

API diagnostics are stored under `artifacts/wgo-ego-demo/api-diagnostics/`,
with individual request bodies and original responses (credentials redacted).
`summary.json` indexes the tests. Tests confirmed the project-local key and
model access, successful synthetic-image inference, and acceptance of individual
generation settings. Two minimal text requests reported HTTP 402 for depleted
prepayment credits before payment setup; after payment setup, minimal text and
synthetic-image requests succeeded. A captured original annotation response
separately reported HTTP 200 with `PROHIBITED_CONTENT`, no candidates, and
response ID `8vTJasLsI62sz7IP6ZvfyAc`. The response does not identify a particular
frame or filter rule, so the exact content trigger remains undetermined.
The full pipeline was retried after payment setup in
`runs/20261010-081904-410652`; segmentation and its built-in retry were still
blocked with `PROHIBITED_CONTENT`. Payment setup did not resolve that block.

The runner loads configuration from the project root rather than from the
current working directory. Its Python base runtime and dependencies are inside
the project. Temporary files, media cache and worker output also remain inside
the project. The sibling dataset directory is the only data location outside
`refiner`.

The pipeline converts one local-video row to a robotics row, runs
`mdr.robotics.subtask_annotation` followed by `mdr.robotics.subtask_labeling`, and
writes JSONL through the local launcher. Preparation uses the original
timestamped-contact-sheet function: 0.5-second sampling, 224px tile width,
20 frames per sheet and five columns.

The source dataset is
[macrodata/WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), pinned
to revision `9cfb7bb98d60a58319f4b3d7b41374eedfb7980c`. Only the first episode's
video column and small Parquet metadata ranges were downloaded, approximately
25 MB rather than the complete 1.4 GB dataset.
