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
from utils.optimizer import SAHIOptimizer

# constants and functions -------------------------
STUDY_ID = "study1"
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
LS_TEST = ["test_a01", "test_a02", "test_a03",
           "test_b01", "test_b02", "test_b03",]
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
    iters = args.iters
    thread = args.thread
    n_samples = args.n_samples
    modelname = args.modelname
    n_params = DICT_PARAMS[modelname]
    total_steps = 4 if args.test else N_STEPS
            
    DIR_DATA = PATHS["DIR_DATA"] / STUDY_ID
    FILE_OUT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"results_{thread}.csv"
    MEM_OUT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"memory_{thread}.csv"
    DIR_PROJECT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"thread_{thread}" / f"{modelname}_{n_samples}"

    # reset torch memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # log ---------------------------
    line_shared = dict({
        "model": modelname,
        "n_params": n_params,
        "n_samples": n_samples,
        "thread": thread,
        "iters": iters,
    })

    # data ------------------------
    seed = string_to_seed(f"{iters}_{thread}")
    random.seed(seed)

    data = YOLO_API(DIR_DATA)
    data.shuffle_train_val(split_src="train", n=int(n_samples), suffix=thread)
    for split in LS_TEST:
        data.make_split(split_src=split, suffix=thread)
    path_yaml = data.save_yaml(classes=["ant"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)
        
    # training ------------------------
    epochs, patience = get_config(BATCH, int(n_samples), total_steps=total_steps)
    time_start = time.time()
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
        name=f"iter_{iters}",
    )
    time_passed = time.time() - time_start
    # get number of epochs has elapsed
    n_epoch = model.trainer.epochs
    n_steps_per_epoch = int(n_samples) // BATCH
    sec_per_step = time_passed / (n_epoch * n_steps_per_epoch)
    sec_per_step = round(sec_per_step, 3)
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # convert to MB
    max_mem = round(max_mem, 3)
    
    line = ",".join([str(value) for value in line_shared.values()])
    line += f",{max_mem},{sec_per_step}\n"

    if os.path.exists(MEM_OUT):
        with open(MEM_OUT, "a") as file:
            file.write(line)
    else:
        with open(MEM_OUT, "w") as file:
            file.write(",".join(line_shared.keys()) + ",max_mem_MB,sec_per_step\n")
            file.write(line)
    print("✅ Training completed!")

    # evaluation ------------------------
    data.update_splits()
    for split in LS_TEST:
        test_split = data[split + f"_{thread}"]
        idx_rdm = random.sample(range(len(test_split)), 10)
        obs = test_split.get_detections()
        pils = test_split.get_PILs()
        pils_batch = [pils[i] for i in idx_rdm]
        
        # 1. single image prediction
        optimizer = SAHIOptimizer(model_path=model.trainer.best)
        line_results = optimizer.inference(pils=pils, obs=obs, no_slices=True)
        write_eval(line_shared, line_results, 
                   splitname=split, strategy="baseline", file_out=FILE_OUT)

        # 2. Bayesian optimization - exploration
        optimizer = SAHIOptimizer(model_path=model.trainer.best)
        optimizer.bo_optimize("count", pils=pils_batch, xi=5)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results, 
                   splitname=split, strategy="bo_exploration", file_out=FILE_OUT)

        # 3. Bayesian optimization - exploitation
        optimizer = SAHIOptimizer(model_path=model.trainer.best)
        optimizer.bo_optimize("count", pils=pils_batch, xi=1)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results, 
                   splitname=split, strategy="bo_exploitation", file_out=FILE_OUT)

        # 4. grid search
        optimizer = SAHIOptimizer(model_path=model.trainer.best)
        optimizer.grid_optimize("count", pils=pils_batch)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results,
                     splitname=split, strategy="grid_search", file_out=FILE_OUT)

        print(f"✅ Evaluation {split} completed!")
    
    if iters != "0":
        shutil.rmtree(DIR_PROJECT / f"iter_{iters}" / "weights", ignore_errors=True)

def write_eval(
    line_shared,
    line_results,
    splitname,
    strategy,
    file_out,
):
    line_out = line_shared.copy()
    line_out["split"] = splitname
    line_out["strategy"] = strategy
    line_out.update(line_results) # add model-specific results

    if os.path.exists(file_out):
        with open(file_out, "a") as file:
            file.write(",".join([str(value) for value in line_out.values()]) + "\n")
    else:
        with open(file_out, "w") as file:
            file.write(",".join(line_out.keys()) + "\n")
            file.write(",".join([str(value) for value in line_out.values()]) + "\n")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iters", type=str, default="0")
    parser.add_argument("-t", "--thread", type=str, default="0")
    parser.add_argument("-n", "--n_samples", type=str, default="32")
    parser.add_argument("-m", "--modelname", type=str, default="yol1o12n")
    parser.add_argument("--test", action="store_true", help="use small number of samples for testing")
    args = parser.parse_args()
    
    # main(args)
    try:
        main(args)
    except Exception as e:
        taskid = f"{STUDY_ID}_{args.modelname}_{args.n_samples}_{args.thread}_{args.iters}"
        errors = str(e)
        path_log = PATHS["DIR_SRC"] / "logs" / "errors" / f"{taskid}.txt"
        # create directory if not exists
        os.makedirs(path_log.parent, exist_ok=True)
        with open(path_log, "w") as file:
            file.write(errors)
        # prevent early-termination of the job
        time.sleep(180)
        print(e)
