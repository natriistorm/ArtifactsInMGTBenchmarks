#!/bin/bash
#SBATCH --job-name=atd-ace
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
# Array size = 4 conditions x 10 subsets x 3 seeds = 120.
# Get the exact number with:  python train_transfer.py --count-tasks --seeds 0 1 2
#SBATCH --array=0-119%16

set -euo pipefail

# --- environment -----------------------------------------------------------
# These nodes have network access, so no prefetch is required. HF_HOME just keeps the
# cache inside the project so array tasks share one copy instead of re-downloading.
export HF_HOME="${HF_HOME:-$PWD/.hf}"
export TOKENIZERS_PARALLELISM=false
# HF_HUB_OFFLINE is deliberately NOT set: these nodes have network access, so the model
# downloads on first use. Only set it if you have run `--prefetch` on a login node first --
# setting it with an empty cache fails outright rather than falling back to the network.

module load python/3.11 cuda 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

mkdir -p logs/slurm

# --- one grid cell per array task ------------------------------------------
# Splits, clusters and bias probes must exist first (CPU only, ~15 min):
#     python train_transfer.py --prepare
# 3 epochs over up to 20k balanced documents, with best-epoch selection on a
# cluster-disjoint dev split. The old 1-epoch/4k setup left mDeBERTa undertrained,
# which made the "a single feature beats the fine-tuned model" comparison unfair.
python train_transfer.py --train \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --seeds 0 1 2 \
    --epochs 3 --n-train 20000 \
    --batch 32 --amp

# When the whole array has finished:
#     python train_transfer.py --report
