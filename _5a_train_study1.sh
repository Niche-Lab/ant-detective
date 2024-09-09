#!/bin/bash
#SBATCH --job-name=ast
#SBATCH --partition=dgx_normal_q
#SBATCH --account=vos
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

source activate tf
source .env

for i in {1..5}
do
    for model in "yolov8n.pt" "yolov8m.pt"
    do
        for n in 64 128 256 512 1024
        do
            python3.9 _4_train.py\
                --model ${model}\
                --study 1\
                --n $n\
                --dir_out $DIR_SRC/out/thread_${SLURM_ARRAY_TASK_ID}\
                --dir_data ${DIR_DATA_STUDY1}_${SLURM_ARRAY_TASK_ID}
        done
    done
done
