#!/usr/bin/env bash
#SBATCH -A bislam
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --job-name="WESAD_EVAL_ECG"

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stamp

# Important: Make sure the cache is clear if you had 0-feature runs before
# rm -rf experiments/features_cache/*

echo "🚀 Starting Full WESAD Evaluation Pipeline"
python -u experiments/eval_classifiers.py