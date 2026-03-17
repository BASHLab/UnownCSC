#!/usr/bin/env bash
#SBATCH -A bislam
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH --job-name="STAMP_FSQ_Base"

module load cuda/11.8.0/4w5kyjs
# 2. Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stamp

# 4. Run the training script (-u forces unbuffered output so the logs update instantly)
python -u src/train_utils/train_wesad.py \
    --resume_path saved_chk_dir/stamp_wesad_ep100.pth \
    --loss hybrid \
    --lr 5e-5 \
    --epochs 50 \
    --quantizer fsq