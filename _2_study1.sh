#!/bin/bash
#SBATCH --job-name=study1
#SBATCH -t 143:59:59
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=128G
#SBATCH --account=niche_squad
#SBATCH --array=0 # Job array
#SBATCH --output=logs/study1_%A_%a.out
#SBATCH --error=logs/study1_%A_%a.out

# Load necessary modules (if required)
module load Miniconda3/24.7.1-0
source activate pyniche

# THREAD=${SLURM_ARRAY_TASK_ID}
MODELS=("yolo11n" "yolo11m" "rtdetr-l")
N_SAMPLES=(64 256 1024)

# Run the script with different configurations
for ITER in {1..100}; do
    for MODEL in "${MODELS[@]}"; do
            for N_SAMPLE in "${N_SAMPLES[@]}"; do
                /home/niche/.conda/envs/pyniche/bin/python study1.py \
                    --thread $THREAD \
                    --iters $ITER \
                    --modelname $MODEL \
                    --n_samples $N_SAMPLE
            done
        done
    done
done

# test run
# /home/niche/.conda/envs/pyniche/bin/python study1.py \
#                     --thread 0 \
#                     --iters 98 \
#                     --modelname "rtdetr-l" \
#                     --n_samples 1024
