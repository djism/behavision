# Roadmap

The lab deliverable comes first. The product work only starts after the four
ITI videos are processed and checked.

## Phase 0: finish and check the four videos

- Process D168, D196, D203 and D303 (D303 is done locally). Each video gets
  `annotated.mp4`, `review/` and per-bout results.
- Check correctness on each video:
  - Watch every inactive bout in the review page and confirm the mouse is
    actually still (not grooming, not hidden).
  - Hand-label two 5-minute windows per video (one calm, one busy) and compare
    them frame by frame with the pipeline.
  - List every untracked or uncertain stretch and the reason for it (buried in
    nesting, out of view, etc.).
- Combine the results with `behavision-combine` and share them with the lab.

**Exit criterion:** the lab agrees the bouts match what they see in the video.

## Phase 1: evaluation

- Build a ground-truth set of 30–60 minutes of labeled video across mice, with
  a short labeling guide so that labels are consistent.
- Measure:
  - Frame-level precision, recall and F1 for "inactive".
  - Bout-level matching (IoU ≥ 0.5) and onset/offset error in seconds.
  - Total inactive time per video, compared with the labels.
  - Agreement between two human labelers, as the ceiling.
- Run ablations: SAM2 model size, motion threshold, minimum bout length,
  mask dilation, and re-detection settings.
- Write up a failure analysis with example clips.
- `behavision-eval` command and a tracked results table, so any change to the
  pipeline is compared against the same labels.

## Phase 2: inference and systems

The method stays the same (see the quality rule below). Only changes that leave
the output unchanged are allowed.

- `torch.inference_mode`, and splitting a video into segments that run on
  several GPUs at once.
- bfloat16 on A100/L4/A10G, accepted only if the Phase 1 metrics don't change.
- Benchmarks of throughput (video-minutes per GPU-hour) on the T4 (Colab,
  Kaggle), A100 (SeaWulf) and Mac MPS.
- One job runner with interchangeable backends (local, Slurm, Colab, Kaggle):
  the same command submits a batch, tracks progress, resumes after a failure
  and collects results, wherever it runs.
- Docker image published free on GitHub Container Registry; the Slurm side
  can run it through Apptainer.
- Continuous integration on GitHub Actions: tests plus a short CPU smoke run
  on a tiny public clip.
- Public demo with no paid servers:
  - "Open in Colab" notebook that runs the pipeline on the user's own GPU
    session.
  - Hugging Face Space (free CPU) that browses precomputed results on public
    clips.
  - Project page on GitHub Pages with metrics, benchmarks and example clips.

**Budget: zero.** Everything runs on SeaWulf, Colab, Kaggle, the Mac and free
hosting.

## Phase 3: generalize

- Describe each use case in config only: object prompt, measurement (motion,
  speed, position, distance to a region) and event rule ("below X for Y s").
- Second domain on public data: stopped vehicles (for example "car stopped for
  more than 30 s in a marked zone"). It gets its own small labeled set and the
  same metrics.
- Goal: adding a domain needs a new config file and labels, not code changes.

## Phase 4: agents

- MCP server for asking questions about results ("compare inactivity in the
  last 10 minutes across all mice").
- Setup agent: from a plain-language description and a few sample frames, it
  drafts the config, runs a preview, checks it against a few labels and
  iterates.
- Evaluation suite for the setup agent: a set of tasks (domain, description,
  labels) and a score for how often the agent's config meets the target
  metrics.

## Rules

- **Quality over speed.** Keep Grounding DINO + SAM2. No faster method that
  changes results.
- **Data.** Lab videos and results stay private unless the PI approves. Public
  demos use public data.
