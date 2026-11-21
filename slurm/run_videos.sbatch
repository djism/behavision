#!/bin/bash
# One array task per video. Submit from the repository root:
#   sbatch --array=1-$(wc -l < videos.txt)%4 slurm/run_videos.sbatch videos.txt /path/to/results
# videos.txt: one absolute video path per line. %4 caps concurrent GPU jobs.
# SeaWulf a100 nodes: 4 A100s, 64 CPUs, 256 GB each; one GPU per video.
#SBATCH --job-name=behavision
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
LIST=${1:?video list file}
RESULTS=${2:?results root}
CONFIG=${CONFIG:-configs/iti_homecage.yaml}

VIDEO=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$VIDEO" ] || { echo "No line ${SLURM_ARRAY_TASK_ID} in $LIST"; exit 1; }
NAME=$(basename "${VIDEO%.*}")

# Environment created once on a login node (see docs/inactivity_pipeline.md);
# behavision_env.sh lives in the repository root (not committed).
source "${BEHAVISION_ENV_SETUP:-$PWD/behavision_env.sh}"
export HF_HUB_OFFLINE=1  # weights were downloaded in advance

# Temporary chunk frames go to node-local scratch; results go to shared storage.
WORK=${TMPDIR:-/tmp}/behavision_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf "$WORK"' EXIT

echo "Task ${SLURM_ARRAY_TASK_ID}: $VIDEO on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
behavision-run "$VIDEO" --config "$CONFIG" --out "$RESULTS/$NAME" --work-dir "$WORK"
