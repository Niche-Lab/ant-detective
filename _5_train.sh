source .env

for i in {1..25}
do
    for n in 64 128 256 512 1024
    do
        python _5_train.py\
            --model yolov8m.pt\
            --n $n\
            --dir_out $DIR_SRC/out/thread_$1\
            --dir_data ${DIR_DATA_YOLO}_$1
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

