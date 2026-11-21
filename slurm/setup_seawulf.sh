#!/bin/bash
# One-time SeaWulf setup. Run on a login node from the repository root:
#   bash slurm/setup_seawulf.sh
# Everything (conda env, pip/conda caches, model weights) goes under BASE on
# project storage, because home directories are small (20 GB) and easily full.
set -euo pipefail
BASE=${BASE:-/gpfs/projects/ShresthaGroup/behavision}
ENV=$BASE/env
mkdir -p "$BASE"/{hf_cache,cache,conda_pkgs,mpl}

export CONDA_PKGS_DIRS=$BASE/conda_pkgs PIP_CACHE_DIR=$BASE/cache/pip XDG_CACHE_HOME=$BASE/cache
export HF_HOME=$BASE/hf_cache MPLCONFIGDIR=$BASE/mpl
module load anaconda/3
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
source activate "$ENV" 2>/dev/null || conda activate "$ENV"

# CUDA wheels bundle their own CUDA runtime; only the GPU driver is needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
SAM2_BUILD_CUDA=0 pip install -e ".[tracking]"

# Download model weights now: compute nodes run offline.
python - <<'EOF'
from behavision.detection.grounding_dino import GroundingDINODetector
from behavision.segmentation.sam2_video import SAM2VideoSegmenter
GroundingDINODetector(device="cpu"); SAM2VideoSegmenter(device="cpu")
print("Model weights cached")
EOF

cat > behavision_env.sh <<EOF
module load anaconda/3
source activate $ENV 2>/dev/null || conda activate $ENV
export HF_HOME=$BASE/hf_cache MPLCONFIGDIR=$BASE/mpl XDG_CACHE_HOME=$BASE/cache
EOF
mkdir -p logs
echo "Setup done. Environment file: $PWD/behavision_env.sh"
