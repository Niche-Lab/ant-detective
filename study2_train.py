# native
import os
import sys
import time
import random
import argparse
import hashlib
import shutil
import numpy as np

# 3-party
from ultralytics import YOLO, RTDETR
import torch

# local imports
from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API

# constants and functions -------------------------
STUDY_ID = "study2"
DICT_PARAMS = dict({
    "rtdetr-l": 45, # 53.0
    "rtdetr-x": 86, # 54.8
    "yolo12n": 2.6, # 40.6
    "yolo12m": 20.2,
    "yolo12x": 59.1, # 55.2
    "yolo11n": 2.6, # 39.5
    "yolo11m": 20.1,
    "yolo11x": 56.9, # 54.7
})
BATCH = 16  # default batch size for training
N_STEPS = 8192  # total number of training steps. use "4" for testing purposes

def string_to_seed(s):
    # Use hashlib to get a consistent integer from a string
    hash_object = hashlib.md5(s.encode())  # can also use sha256
    seed_int = int(hash_object.hexdigest(), 16) % (2**32)
    return seed_int

def get_config(batch, n, total_steps=N_STEPS):
    steps_per_epoch = n // batch
    epochs = total_steps // steps_per_epoch
    patience = epochs // 4
    return epochs, patience

def main(args):
    modelname = args.modelname
    thread = args.thread
    total_steps = 4 if args.test else N_STEPS
            
    DIR_DATA = PATHS["DIR_DATA"] / STUDY_ID
    DIR_PROJECT = PATHS["DIR_SRC"] / "out" / STUDY_ID / modelname

    # data ------------------------
    data = YOLO_API(DIR_DATA)
    data.shuffle_train_val(split_src="train", suffix=thread)
    path_yaml = data.save_yaml(classes=["ant"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)
        
    # training ------------------------
    epochs, patience = get_config(BATCH, 1024, total_steps=total_steps)
    model.train(
        # data
        data=path_yaml,
        batch=BATCH,
        # check ultralytics/data/augment.py line 1153
        # s = random.uniform(1, 1 + self.scale)
        scale=0.9, # [1, 1 + scale]
        flipud=0.5, fliplr=0.5, # horizontal and vertical flip
        # training
        epochs=epochs,
        patience=patience,
        workers=4,
        # output: DIR_PROJECT/iter_{iters}/
        project=DIR_PROJECT,
        name=".",
    )
    print("✅ Training completed!")

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelname", type=str, default="yolo11n")
    parser.add_argument("-t", "--thread", type=str, default="0")
    parser.add_argument("--test", action="store_true", help="use small number of samples for testing")
    args = parser.parse_args()
    
    main(args)
