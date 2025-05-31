# native
import os
import sys
import time
import random
import argparse
import hashlib
import shutil

# 3-party
from ultralytics import YOLO, RTDETR
import torch

# local imports
from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API
from pyniche import evaluate
from utils.optimizer import SAHIOptimizer, predict_sahi

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
N_SLICES = [(4, 10), (4, 5), (2, 5), 
            (4, 4), (2, 4), (4, 2), (2, 2)]

def string_to_seed(s):
    # Use hashlib to get a consistent integer from a string
    hash_object = hashlib.md5(s.encode())  # can also use sha256
    seed_int = int(hash_object.hexdigest(), 16) % (2**32)
    return seed_int

def main(args):
    iters = args.iters
    thread = args.thread
    modelname = args.modelname
    n_params = DICT_PARAMS[modelname]
    is_finetune = "2_FT" if args.finetune else "1_NOFT"
            
    DIR_DATA = PATHS["DIR_DATA"] / STUDY_ID
    FILE_OUT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"results_{thread}.csv"
    DIR_PROJECT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"thread_{thread}" / f"{modelname}_{is_finetune}"
    # log ---------------------------
    line_shared = dict({
        "model": modelname,
        "n_params": n_params,
        "is_finetune": is_finetune,
        "thread": thread,
        "iters": iters,
    })

    # data ------------------------
    seed = string_to_seed(f"{iters}_{thread}")
    random.seed(seed)

    data = YOLO_API(DIR_DATA)    
    split_dense = data["test_dense"]
    n_dense = len(split_dense)
    idx_train = random.sample(range(n_dense), 1)
    idx_test = set(range(n_dense)) - set(idx_train)

    if args.finetune:
        data.slice_split(
            split_src="test_dense", 
            split_dst=f"val_dense_{thread}", 
            n_slices=N_SLICES,
            ls_idx=idx_train,)
        data.make_split(
            split_src="train",
            suffix=f"{thread}",            
        )
        data.make_split(
            split_src=f"val_dense_{thread}",
            split_dst="val",
            suffix=f"{thread}",
        )
    else:
        data.shuffle_train_val(
            split_src="train",
            k=10,
            suffix=f"{thread}")
    data.make_split(
        split_src="test_dense", 
        split_dst="test", 
        ls_idx=idx_test,
        suffix=f"{thread}")   
    path_yaml = data.save_yaml(classes=["ant"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)
    
    # training ------------------------
    epochs = 1 if args.test else 300  # set epochs to 1 for testing
    patience = 1 if args.test else 100  # set patience to 1 for testing
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
    print("✅ Training completed!")

    # evaluation ------------------------
    path_model_eval = model.trainer.best

    data.update_splits()
    split_test = data[f"test_{thread}"]
    obs = split_test.get_detections()
    pils = split_test.get_PILs()

    obs_hptune  = split_dense.get_detections()[idx_train[0]]
    pils_hptune = split_dense.get_PILs()[idx_train[0]]
    
    # 1. single image prediction
    optimizer = SAHIOptimizer(model_path=path_model_eval)
    line_results = optimizer.inference(pils=pils, obs=obs, no_slices=True)
    write_eval(line_shared, line_results, 
                metric="baseline", strategy="baseline", file_out=FILE_OUT)

    # 2. Bayesian optimization - exploration
    optimizer = SAHIOptimizer(model_path=path_model_eval)
    optimizer.bo_optimize("count", pils=pils_hptune, xi=70)
    line_results = optimizer.inference(pils=pils, obs=obs)
    write_eval(line_shared, line_results, 
                metric="count", strategy="bo_exploration", file_out=FILE_OUT)

    # 3. Bayesian optimization - exploitation
    optimizer = SAHIOptimizer(model_path=path_model_eval)
    optimizer.bo_optimize("count", pils=pils_hptune, xi=7)
    line_results = optimizer.inference(pils=pils, obs=obs)
    write_eval(line_shared, line_results, 
                metric="count", strategy="bo_exploitation", file_out=FILE_OUT)

    # 4. grid search
    optimizer = SAHIOptimizer(model_path=path_model_eval)
    optimizer.grid_optimize("count", pils=pils_hptune)
    line_results = optimizer.inference(pils=pils, obs=obs)
    write_eval(line_shared, line_results,
                metric="count", strategy="grid_search", file_out=FILE_OUT)

    if args.finetune:
        # 2b. Bayesian optimization - exploration with finetuning
        optimizer = SAHIOptimizer(model_path=path_model_eval)
        optimizer.bo_optimize("map", obs=obs_hptune, pils=pils_hptune, xi=0.1)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results,
                    metric="map", strategy="bo_exploration", file_out=FILE_OUT)
        # 3b. Bayesian optimization - exploitation with finetuning
        optimizer = SAHIOptimizer(model_path=path_model_eval)
        optimizer.bo_optimize("map", obs=obs_hptune, pils=pils_hptune, xi=0.01)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results,
                    metric="map", strategy="bo_exploitation", file_out=FILE_OUT)
        # 4b. grid search with finetuning
        optimizer = SAHIOptimizer(model_path=path_model_eval)
        optimizer.grid_optimize("map", obs=obs_hptune, pils=pils_hptune)
        line_results = optimizer.inference(pils=pils, obs=obs)
        write_eval(line_shared, line_results,
                    metric="map", strategy="grid_search", file_out=FILE_OUT)
        
    print("✅ Evaluation completed!")
    
    if iters != "0":
        shutil.rmtree(DIR_PROJECT / f"iter_{iters}" / "weights", ignore_errors=True)

def write_eval(
    line_shared,
    line_results,
    metric,
    strategy,
    file_out,
):
    line_out = line_shared.copy()
    line_out["metric"] = metric
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
    parser.add_argument("-m", "--modelname", type=str, default="yol1o12n")
    parser.add_argument("--finetune", action="store_true", help="further finetuning")
    parser.add_argument("--test", action="store_true", help="use small number of samples for testing")
    args = parser.parse_args()
    
    if args.test:
        main(args)
    else:
        try:
            main(args)
        except Exception as e:
            print(e)
            taskid = f"{STUDY_ID}_{args.modelname}_{args.finetune}_{args.thread}_{args.iters}"
            errors = str(e)
            path_log = PATHS["DIR_SRC"] / "logs" / "errors" / f"{taskid}.txt"
            # create directory if not exists
            os.makedirs(path_log.parent, exist_ok=True)
            with open(path_log, "w") as file:
                file.write(errors)
            # prevent early-termination of the job
            time.sleep(180)
