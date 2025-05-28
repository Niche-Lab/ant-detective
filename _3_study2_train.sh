#!/bin/bash
#SBATCH --job-name=study2_train
#SBATCH -t 143:59:59
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=128G
#SBATCH --account=niche_squad
#SBATCH --array=0-2 # Job array for different models
#SBATCH --output=logs/study2_%A_%a.out
#SBATCH --error=logs/study2_%A_%a.out

# Load necessary modules (if required)
module load Miniconda3/24.7.1-0
source activate pyniche

THREAD=${SLURM_ARRAY_TASK_ID}
# Define models, configs, and sample sizes
MODELS=("yolo11n" "yolo11m" "rtdetr-l")

# if job=0, yolo11n, if job=1, yolo11m, if job=2, rtdetr-l

if [ $THREAD -eq 0 ]; then
    export MODEL_NAME="yolo11n"
elif [ $THREAD -eq 1 ]; then
    export MODEL_NAME="yolo11m"
elif [ $THREAD -eq 2 ]; then
    export MODEL_NAME="rtdetr-l"
else
    echo "Invalid thread ID: $THREAD"
    exit 1
fi

/home/niche/.conda/envs/pyniche/bin/python\
 study2_train.py\
  --thread $THREAD\
  --modelname $MODEL_NAME
