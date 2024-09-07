"""
Training YOLOv8

This trial aims to compare the performance difference based on two factors:
    1. the number of images in the dataset
        - 32
        - 64
        - 128
        - 256
        - 512
        
    2. size of the model
        - YOLOv8n: mAP 0.5:0.95 = 37.3; params = 3.2M;
        - YOLOv8m: mAP 0.5:0.95 = 50.2; params = 25.9M;
        - YOLOv8x: mAP 0.5:0.95 = 53.9; params = 68.2M;
"""

# 1. Imports -------------------------------------------------------------------
# native imports
import os
import argparse

# import torch
# Limit to 50% of the available memory
# torch.cuda.set_per_process_memory_fraction(0.2, device=0)

from pyniche.trainer import NicheTrainer
from pyniche.models.detection.yolo import NicheYOLO

# 2. Global Variables ----------------------------------------------------------
DEVICE = "cuda"


def main(args):
    # extract arguments
    model = args.model
    n = int(args.n)
    study = int(args.study)
    DIR_DATA_ROOT = args.dir_data
    DIR_OUT_ROOT = args.dir_out

    # 1. Set up directories -------------------------------------------------------------
    DIR_DATA = os.path.join(
        DIR_DATA_ROOT,
    )
    FILE_OUT = os.path.join(
        DIR_OUT_ROOT,
        "results.csv",
    )
    # resuem iteration
    i = 1
    while True:
        if study == 1:
            DIR_OUT = os.path.join(
                DIR_OUT_ROOT,
                "study%d_%d_%d" % (study, n, i),
            )
        else:
            DIR_OUT = os.path.join(
                DIR_OUT_ROOT,
                "study%d_%d" % (study, i),
            )
        if not os.path.exists(DIR_OUT):
            break
        i += 1
    # e.g., yolov8n_32_2: model, n, iteration
    os.makedirs(DIR_OUT, exist_ok=True)

    # 3. Initialize outputs -------------------------------------------------------------
    if not os.path.exists(FILE_OUT):
        os.makedirs(os.path.dirname(FILE_OUT), exist_ok=True)
        with open(FILE_OUT, "w") as file:
            file.write(
                "map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,study,split,model,n\n"
            )

    # 4. Modeling -------------------------------------------------------------
    trainer = NicheTrainer(device=DEVICE)
    trainer.set_model(
        modelclass=NicheYOLO,
        checkpoint=model,
    )
    trainer.set_data(
        dataclass=DIR_DATA,
        batch=16,
        n=n if study == 1 else None, # use all data for study 2
        k=5 if study == 1 else 0, # no shuffle for study 2
        classes=["ant"],
    )
    trainer.set_out(DIR_OUT)
    trainer.fit(
        epochs=100,
        rm_threshold=0,
        copy_paste=0.3,
        mixup=0.15,
    )

    # 5. Evaluation -------------------------------------------------------------
    if study == 1:
        test_splits = ["test_a01", "test_a02", "test_a03", 
                       "test_b01", "test_b02", "test_b03",]
    else:
        test_splits = ["test", "test_2x2", "test_2x4", "test_4x4", "test_4x5", ]
    for test_split in test_splits:
        metrics = trainer.evaluate_on_test(
            split=test_split,
            name_task=DIR_OUT + "_" + test_split,
            conf=0.5,)
        metrics["study"] = study
        metrics["split"] = test_split
        metrics["model"] = model.split(".")[0]  # remove .pt
        metrics["n"] = n
        line = ",".join([str(value) for value in metrics.values()])
        with open(FILE_OUT, "a") as file:
            file.write(line + "\n")

    # remove model weights
    os.remove(os.path.join(DIR_OUT, "weights", "best.pt"))
    os.remove(os.path.join(DIR_OUT, "weights", "last.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="yolo checkpoint",
    )
    parser.add_argument(
        "--study",
        type=int,
        help="study number, 1 or 2",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of images in training set",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--dir_data",
        type=str,
        help="data directory",
    )
    args = parser.parse_args()
    main(args)
