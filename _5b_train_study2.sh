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

source activate tf
source .env

for model in "yolov8n.pt" "yolov8m.pt"
do
    python3.9 _4_train.py\
        --model ${model}\
        --study 2\
        --n 0\
        --dir_out $DIR_SRC/out/study2\
        --dir_data ${DIR_DATA_STUDY2}
done

# 9c: VRAM: batch 16, 640x640: 11.2 G
# 9e: VRAM: batch 16, 640x640: 24.4 G
# 8n: VRAM: batch 16, 640x640: 2.8 G
# 8m: VRAM: batch 16, 640x640: 7.23 G
# 8x: VRAM: batch 16, 640x640: 13.5 G


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

