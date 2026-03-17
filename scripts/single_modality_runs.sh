#!/usr/bin/env bash

# Define the 10 isolated WESAD modalities
MODALITIES=(
    "chest_ACC" "chest_ECG" "chest_EMG" "chest_RESP" "chest_EDA" "chest_TEMP"
    "wrist_ACC" "wrist_BVP" "wrist_EDA" "wrist_TEMP"
)

# Loop through each modality and dynamically submit a dedicated SLURM job
for MOD in "${MODALITIES[@]}"; do
    echo "Submitting Phase 1 (MSE) and Phase 2 (Soft-DTW) for: $MOD"

    # The 'sbatch <<EOT' command feeds everything up to 'EOT' directly to the SLURM scheduler
    sbatch <<EOT
#!/usr/bin/env bash
#SBATCH -A bislam
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH --job-name="STAMP_${MOD}"


module load cuda/11.8.0/4w5kyjs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stamp

echo "====================================================="
echo " Phase 1: Base Training (MSE) for \$MOD "
echo "====================================================="
python -u src/train_utils/train_wesad_single_modality.py \\
    --modality $MOD \\
    --quantizer fsq \\
    --epochs 100 \\
    --batch_size 32 \\
    --loss mse

echo "====================================================="
echo " Phase 2: Fine-Tuning (Hybrid Soft-DTW) for \$MOD "
echo "====================================================="
# Notice how we dynamically route the resume_path to the exact modality folder created in Phase 1
python -u src/train_utils/train_wesad_single_modality.py \\
    --modality $MOD \\
    --resume_path saved_chk_dir_single/${MOD}/stamp_${MOD}_ep100.pth \\
    --loss hybrid \\
    --lr 5e-5 \\
    --epochs 50 \\
    --quantizer fsq

echo "✅ Full Pipeline complete for $MOD"
EOT
done

echo "🎉 All 10 jobs have been submitted to the SLURM queue!"