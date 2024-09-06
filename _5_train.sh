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
    for model in "yolov8m.pt" "yolov8x.pt"
    do
        for n in 64 128 256 512 1024
        do
            python3.9 _5_train.py\
                --model $model\
                --n $n\
                --dir_out $DIR_SRC/out/thread_${SLURM_ARRAY_TASK_ID}\
                --dir_data ${DIR_DATA_YOLO}_${SLURM_ARRAY_TASK_ID}
        done
    done
done


# for i in {1..300}
# do
#     for model in "yolov9c.pt" "yolov9e.pt" "yolov8n.pt" "yolov8m.pt" "yolov8x.pt"
#     do
#         for n in 64 128 256 512 1024
#         do
#             python _2_yolo.py\
#                 --model $model\
#                 --config $1\
#                 --n $n\
#                 --dir_out $dir_out\
#                 --dir_data $dir_data
#         done
#     done

# done

