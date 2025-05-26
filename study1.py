# native
import os
import sys
import random
import argparse
import hashlib
import shutil

# 3-party
from ultralytics import YOLO, RTDETR

# local imports
from paths import PathFinder
from evaluate import eval_metrics
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API

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
           "test_b01", "test_b02", "test_b03"]
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
    DIR_PROJECT = PATHS["DIR_SRC"] / "out" / STUDY_ID / f"thread_{thread}" / f"{modelname}_{n_samples}"

    # data ------------------------
    seed = string_to_seed(f"{iters}_{thread}")
    random.seed(seed)

    data = YOLO_API(DIR_DATA)
    data.shuffle_train_val(split_src="train", n=int(n_samples), suffix=thread)
    for split in LS_TEST:
        data.make_split(split_src=split, suffix=thread)
    path_yaml = data.save_yaml(classes=["cow"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)
        
    # training ------------------------
    epochs, patience = get_config(BATCH, int(n_samples), total_steps=total_steps)
    model.train(
        # data
        data=path_yaml,
        batch=BATCH,
        scale=0.9, # [1 - scale, 1 + scale]
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
    for split in LS_TEST:
        out = model.val(
            data=path_yaml,
            split=split,
            conf=0.25,
            project=DIR_PROJECT.as_posix() + f"-{split}",
            name=f"iter_{iters}",
        )
        metrics = eval_metrics(out)
        str_profile = f"{modelname},{n_params},{n_samples},{thread},{iters},{split},"
        str_metrics = ",".join([str(value) for value in metrics.values()])
        line = str_profile + str_metrics
        if os.path.exists(FILE_OUT):
            with open(FILE_OUT, "a") as file:
                file.write(line + "\n")
        else:
            with open(FILE_OUT, "w") as file:
                file.write("model,n_params,n_samples,thread,iters,split," + ",".join(metrics.keys()) + "\n")
                file.write(line + "\n")
        shutil.rmtree(DIR_PROJECT / f"iter_{iters}" / "weights", ignore_errors=True)
        print(f"✅ Evaluation {split} completed!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iters", type=str, default="0")
    parser.add_argument("-t", "--thread", type=str, default="0")
    parser.add_argument("-n", "--n_samples", type=str, default="32")
    parser.add_argument("-m", "--modelname", type=str, default="yol1o12n")
    parser.add_argument("--test", action="store_true", help="use small number of samples for testing")
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        taskid = f"{STUDY_ID}_{args.modelname}_{args.n_samples}_{args.thread}_{args.iters}"
        errors = str(e)
        path_log = PATHS["DIR_SRC"] / "log" / "errors" / f"{taskid}.txt"
        # create directory if not exists
        os.makedirs(path_log.parent, exist_ok=True)
        with open(path_log, "w") as file:
            file.write(errors)
        # prevent early-termination of the job
        # time.sleep(180)
        print(e)