# Local stillness validation pilot

This standalone workflow classifies **manually selected windows**, without running
Grounding DINO, SAM2, tracking, MCP, or the chat assistant. It uses OpenCV (already
a project dependency) and Python's standard library to call local Ollama. No paid
API, DeepLabCut, PyTorch, or Transformers installation is needed for this pilot.

## Model and scope

Start with `qwen3-vl:4b`, the approximately 3.3 GB quantized Ollama package of
Qwen3-VL. Runtime memory exceeds the weight download size and grows with frame
count/resolution. Use an up-to-date Ollama (the model page specifies >=0.12.7).
The existing `qwen3:4b` chat model is a different model and is not used here.

Sources: [Ollama package](https://ollama.com/library/qwen3-vl:4b),
[Apache-2.0 model card and Transformers implementation](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[Ollama image API](https://docs.ollama.com/capabilities/vision),
[structured responses](https://docs.ollama.com/capabilities/structured-outputs).

Ollama receives chronological JPEG images with their timestamps, **not a native
video stream**. Default sampling is 8 frames/second, including the first and last
eligible frames of each window; the longest image side is capped at 768 pixels
without upscaling. Inspect saved images: a mouse that is too small, obscured, or
blurred may require a higher `--max-side` or a manually cropped input video.
No automatic crop is inferred. Sparse images can miss fast head/paw movements.
The model has not been validated for laboratory mouse inactivity.

STILL means no observable voluntary action across the sampled window. Grooming,
sniffing, head movement, walking/running, turning, rearing, scratching,
eating/licking, and clear posture adjustment are ACTIVE even without centroid
displacement. Any clear active moment makes the window ACTIVE. Tiny involuntary
movement or noise can be ignored. Uncertain movement or insufficient visual
evidence should produce AMBIGUOUS. This is an observable-behavior proxy; it does
not establish intent, sleep, freezing, or biological ground truth.

## Prepare about 20 human-labeled windows

Copy `configs/stillness_clips.example.csv` to `configs/stillness_clips.csv`.
Replace the placeholder paths/times and add approximately 20 rows. Do not run
the example unchanged. Columns:

| Column | Meaning |
| --- | --- |
| `clip_id` | Unique letters/digits/underscore/hyphen identifier |
| `video_path` | Absolute path, or relative to the CSV's directory |
| `start_s`, `end_s` | Numeric seconds in that video, start included/end excluded |
| `expected_label` | Human `STILL`, `ACTIVE`, or `AMBIGUOUS`; blank permits unlabeled inference |
| `notes` | Optional human annotation; never sent to the model |

Select 2–3 second windows initially. Each row can reference the original long
video (only selected frames are decoded), or an already extracted short clip
(use its own local times). Outputs use the referenced video's timeline; for an
extracted clip, they do not recover its original long-video offset. Windows
must be <=10 seconds; a run accepts 1–30 rows to keep this a small pilot.
Timestamps use frame index divided by the video's reported FPS, as in the
existing reader. Use constant-frame-rate input; variable-frame-rate material
needs timestamp-aware decoding or conversion before relying on these times.

Watch the original clips continuously, ideally with a second human reviewer,
and label them **before** inference. Include convincing STILL clips, each listed
active behavior especially sniffing/grooming/head-only movement, and difficult
visibility/transition cases. Do not select only easy cases or use centroid
stationarity as the human label. Any observed active moment makes a window
ACTIVE. Resolve uncertain human labels as AMBIGUOUS.

## Exact commands from the repository root

Use the Python environment you already use for Behavision, or create one:

```bash
python3 -m venv .venv-stillness
source .venv-stillness/bin/activate
python -m pip install -e .
cp configs/stillness_clips.example.csv configs/stillness_clips.csv
```

Edit `configs/stillness_clips.csv` with your selected paths, windows, and labels.
Install Ollama separately if needed, then start the local Ollama app or run
`ollama serve` in another terminal. Download the local weights once:

```bash
ollama pull qwen3-vl:4b
```

First check paths, decoding, and saved frame visibility without inference:

```bash
python scripts/validate_stillness.py --manifest configs/stillness_clips.csv --output-dir data/outputs/stillness_preview --dry-run
```

Run classification:

```bash
python scripts/validate_stillness.py --manifest configs/stillness_clips.csv --output-dir data/outputs/stillness_validation --model qwen3-vl:4b
```

Use a new output directory for every run. Existing directories are refused.
To preserve more image detail, add `--max-side 1024` (higher memory and latency).
`--sample-fps` can reduce sampling but is capped at 8 for this pilot. At native
video FPS below the requested rate, each available frame is used at most once.
The endpoint is restricted to `127.0.0.1`; `--port` can select a different local
Ollama server. The script requires an already installed model and never pulls
weights automatically. Use local model tags, not cloud models. Keep Ollama in
local-only operation; downloads require internet but inference needs no API key.

## Review the results

- `predictions.csv`: clip ID, video, start/end/duration seconds, human label,
  prediction, reason, status, error, elapsed seconds. Easy to inspect in a sheet.
- `predictions.jsonl`: the same information plus sampled timestamps, human notes,
  and raw model text, flushed after each clip to preserve progress.
- `summary.json`: confusion matrix (human rows/model columns), agreement, STILL
  precision/recall, error count, and false-STILL clip IDs. Missing denominators
  are `null`. Unlabeled rows are excluded from agreement metrics. Errors have a
  separate column; they are never converted to behavioral AMBIGUOUS labels.
- `config.json`: prompt/schema, sampling settings, generation settings, manifest
  hash, installed model metadata including digest. Temperature/seed are fixed;
  identical outputs across hardware/runtime versions are not guaranteed.
- `frames/<clip_id>/`: exactly the JPEGs sent to the model, named by frame index.

Dry-run output intentionally has no behavioral predictions or agreement metrics.
A clip error is saved and processing continues; any errors give exit code 1.
Setup errors give exit code 2. Review every false-STILL prediction, AMBIGUOUS
case, and error against the continuous original video. Reasons are model claims,
not verified evidence. Twenty selected examples are a feasibility check, not an
accuracy estimate for the hour-long video; use a larger held-out set before
scientific use. No acceptance threshold is assumed here.

## Later SeaWulf step (not implemented)

This CLI is noninteractive, uses configurable input/output paths, and processes
one bounded clip at a time. A future Slurm job can run an Ollama server on the
allocated compute node and point this client to its local port, subject to
cluster software/GPU policy. Do not run the hour-long inference on a login node.
Stage weights in advance if compute nodes lack network access. Hardware,
available GPU memory, and supported runtimes still need to be checked on SeaWulf.
The same model family can alternatively use Transformers on CUDA, but that
backend is not part of this pilot and must be validated separately.

Full-video overlapping window generation and merging are intentionally deferred.
Selected pilot windows may be widely separated; merging them would imply evidence
for unseen intervals. A later merger must preserve gaps, let ACTIVE/AMBIGUOUS
evidence veto overlapping STILL coverage, and report window-scale boundary
uncertainty. Pilot outputs are window judgments, not precise inactivity bouts.

Run the offline tests with:

```bash
python -m unittest discover -s tests -p 'test_stillness_validation.py' -v
```
