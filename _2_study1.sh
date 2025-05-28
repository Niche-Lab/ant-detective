#!/bin/bash
#SBATCH --job-name=study1
#SBATCH -t 143:59:59
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=niche_squad
#SBATCH --array=0-4 # Job array
#SBATCH --output=logs/study1_%A_%a.out
#SBATCH --error=logs/study1_%A_%a.out

# Load necessary modules (if required)
source activate pyniche

THREAD=${SLURM_ARRAY_TASK_ID}
# Define models, configs, and sample sizes
MODELS=("rtdetr-l" "yolo11n" "yolo11m")
N_SAMPLES=(64 256 1024)

# Run the script with different configurations
for ITER in {1..100}; do
    for MODEL in "${MODELS[@]}"; do
            for N_SAMPLE in "${N_SAMPLES[@]}"; do
                python study1.py \
                    --thread $THREAD \
                    --iters $ITER \
                    --modelname $MODEL \
                    --n_samples $N_SAMPLE
            done
        done
    done
done


python study1.py\
    --iters 0\
    --thread 1\
    --n_samples 64\
    --modelname yolo11n\
    --test

