#!/bin/bash
#SBATCH --job-name=study2
#SBATCH -t 143:59:59
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=128G
#SBATCH --account=niche_squad
#SBATCH --array=0 # Job array for different models
#SBATCH --output=logs/study2_%A_%a.out
#SBATCH --error=logs/study2_%A_%a.out

# Load necessary modules (if required)
module load Miniconda3/24.7.1-0
source activate pyniche

THREAD=${SLURM_ARRAY_TASK_ID}
# Define models, configs, and sample sizes
MODELS=("yolo11n" "yolo11m" "rtdetr-l")

# Run the script with different configurations
for ITER in {1..100}; do
    for MODEL in "${MODELS[@]}"; do
            /home/niche/.conda/envs/pyniche/bin/python study2.py \
                --thread $THREAD \
                --iters $ITER \
                --modelname $MODEL
            /home/niche/.conda/envs/pyniche/bin/python study2.py \
                --thread $THREAD \
                --iters $ITER \
                --modelname $MODEL \
                --finetune
        done
    done
done
