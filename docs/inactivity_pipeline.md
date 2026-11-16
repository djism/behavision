# Inactivity pipeline

Finds periods when a single tracked mouse does nothing visible, for whole
videos, without hand labels.

## Definition

An **inactive bout** is at least `min_bout_s` (default 2 s) of consecutive
frames with no visible movement of the body, head, or limbs. Walking, turning,
rearing, grooming, sniffing, head movement, scratching, eating, and posture
changes are active. Frames where the mouse cannot be judged (lost tracking, mask
much smaller or larger than usual, e.g. hidden in nesting material) are
**uncertain**; they are never counted as inactive and they end a bout.

## Method

1. **Track** the mouse with Grounding DINO (first frame / recovery) and SAM2
   (masks), in chunks of `chunk_size` frames. Each chunk starts from the previous
   chunk's last mask; a confident detection that disagrees replaces it. If the
   mask is empty or of implausible size for `failure_frames` consecutive frames,
   the mouse is re-detected there and SAM2 restarts. Detections mostly inside
   `exclude_regions` (the cable connector) are ignored.
2. **Measure motion** per frame: the fraction of pixels in the slightly dilated
   mouse mask whose blurred grayscale value changed by more than a pixel delta,
   minus the same fraction for background pixels far from the mouse (removes
   flicker and compression changes affecting the whole frame), then a rolling
   median over `smoothing_frames` so a single noisy frame cannot split a bout.
3. **Calibrate per video, no labels.** The pixel delta is `noise_multiplier`
   times the typical 99th percentile of background change (camera noise), as in
   ezTrack. A higher percentile is avoided because the moving tether cable and
   disturbed bedding inflate it. The still/moving threshold is the valley of that video's motion
   histogram (Otsu, log scale), clipped to `motion_threshold_bounds`; set
   `motion_threshold` to a number to fix it instead.
4. **Bouts**: still runs of at least `min_bout_s` are inactive; shorter pauses
   are active.
5. **Sensitivity**: results are recomputed at several threshold scales and
   minimum bout lengths (`sensitivity.csv`) to show how much conclusions depend
   on these choices.

Without labels there is no accuracy estimate. `review/index.html` offers an
optional ~10 minute spot check of 30 sampled clips.

## Outputs (per video directory)

| File | Contents |
| --- | --- |
| `inactive_bouts.csv` | bout id, start/end frame, start/end s, duration, mean motion |
| `timeline.csv` | per frame: time, state (inactive/active/uncertain), motion, position, mask area |
| `summary.json` | inactive/active/uncertain time, inactive fraction of observed time, bout statistics, latency to first bout, per-bin (5 min) values, calibration |
| `sensitivity.csv` | the same headline numbers across threshold scales x minimum bout lengths |
| `qc.json` | tracked fraction, uncertain fraction, chunk seeding counts, warnings |
| `activity_overview.png` | state strip, motion trace with threshold, motion histogram, per-bin inactive fraction |
| `review/index.html` | sampled clips with Correct/Wrong/Can't tell buttons and CSV export |
| `annotated.mp4` | optional (`--annotated-video`): whole range with box and state |
| `tracking.csv`, `tracking_chunks.csv`, `tracking_chunks/` | raw tracking and per-chunk checkpoints |

`behavision-combine RESULTS_ROOT` writes `all_videos_summary.csv` (one row per
video) and `all_bouts.csv` (all bouts, tagged by video).

## Run locally

Apple-silicon GPU (MPS) tracking ran at about 0.7 frames/s in testing (about
14 h for a 1-hour, 10 fps video), so use local runs for previews and short
windows and SeaWulf GPUs for whole videos.

```bash
python -m pip install -e ".[tracking]"
# 1. Check detection and the exclusion box on frames across the video:
behavision-run VIDEO.avi --config configs/iti_homecage.yaml --out data/outputs/NAME --preview
# 2. Short test (first 5 min), then the whole video:
behavision-run VIDEO.avi --config configs/iti_homecage.yaml --out data/outputs/NAME_5min --end-s 300 --annotated-video
behavision-run VIDEO.avi --config configs/iti_homecage.yaml --out data/outputs/NAME
```

An interrupted run continues from the last completed chunk when restarted with
the same `--out`. Re-run only the cheap stages after changing activity or review
settings: `--stages analyze,review`. Tracking settings or range cannot change
inside an existing output directory.

## Run on SeaWulf (to be verified on the cluster)

One-time setup on a login node; partition names, modules, and CUDA versions
must be checked there (`sinfo -s`, `module avail`):

```bash
git clone https://github.com/djism/behavision.git && cd behavision
module load anaconda   # exact module name on SeaWulf to be checked
conda create -y -p $HOME/envs/behavision python=3.11 && conda activate $HOME/envs/behavision
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the GPU driver
pip install -e ".[tracking]"
export HF_HOME=$HOME/hf_cache
python -c "from behavision.detection.grounding_dino import GroundingDINODetector as D; from behavision.segmentation.sam2_video import SAM2VideoSegmenter as S; D(device='cpu'); S(device='cpu')"
cat > $HOME/behavision_env.sh <<EOF
module load anaconda
conda activate $HOME/envs/behavision
export HF_HOME=$HOME/hf_cache
EOF
```

Then set the GPU partition in `slurm/run_videos.sbatch`, list videos (one
absolute path per line) and submit one array task per video:

```bash
mkdir -p logs
ls /path/to/videos/*.avi > videos.txt
sbatch --array=1-$(wc -l < videos.txt)%4 slurm/run_videos.sbatch videos.txt /path/to/results
behavision-combine /path/to/results   # after all tasks finish
```

Use `--array=1` for the first single-video test. A failed or timed-out task can
be resubmitted with `--array=<task id>`; it resumes from its checkpoints.
