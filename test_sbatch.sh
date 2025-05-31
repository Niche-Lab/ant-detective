python study1.py\
    --iters 0\
    --thread 1\
    --n_samples 64\
    --modelname yolo11n\
    --test



/home/niche/.conda/envs/pyniche/bin/python\
 study2_eval.py\
    --iters 0\
    --thread 0\
    --modelname yolo11n\
    --test

/home/niche/.conda/envs/pyniche/bin/python\
 study2_eval.py\
    --iters 0\
    --thread 0\
    --modelname yolo11n\
    --finetune\
    --test

