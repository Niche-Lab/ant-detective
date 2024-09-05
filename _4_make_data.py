"""
This script is used to prepare the data for training YOLOv8:

1. Turn yaml from path-to-split to txt-to-split
2. Merge train and val
3. Clone the folder for 4 times for multi-threading
"""
import os
from dotenv import load_dotenv
from pyniche.data.yolo.API import YOLO_API

load_dotenv(".env")

DIR_DATA_SRC = os.getenv("DIR_DATA_FINAL")
DIR_DATA_YOLO = os.getenv("DIR_DATA_YOLO")
DIR_TRAIN = os.path.join(DIR_DATA_YOLO, "train")
DIR_TEST = os.path.join(DIR_DATA_YOLO, "test")
YAML_OUT = os.path.join(DIR_DATA_YOLO, "data.yaml")


def file_transfer():
    os.system("mkdir -p %s" % DIR_TRAIN)
    os.system("mkdir -p %s" % DIR_TEST)
    # cp src/train to yolo/train
    os.system("cp -r %s/train %s" % (DIR_DATA_SRC, DIR_TRAIN))
    os.system("cp -r %s/val/images/* %s/images/" % (DIR_DATA_SRC, DIR_TRAIN))
    os.system("cp -r %s/val/labels/* %s/labels/" % (DIR_DATA_SRC, DIR_TRAIN))
    for test in ["test_b01", "test_b03", "test_b04", "test_b05", "test_b06", 
                "test_a01", "test_a02", "test_a03"]:
        dir_test = os.path.join(DIR_DATA_YOLO, test)
        os.system("mkdir -p %s" % dir_test)
        os.system("cp -r %s/%s/images %s" % (DIR_DATA_SRC, test, dir_test))
        os.system("cp -r %s/%s/labels %s" % (DIR_DATA_SRC, test, dir_test))
    os.system("cp -r %s/test_b01 %s/test" % (DIR_DATA_SRC, DIR_DATA_YOLO))        


def make_yolo_data():
    api = YOLO_API(DIR_DATA_YOLO)
    api.yaml_to_txt()
    api.merge_train_val()
    api.save_yaml(classes=["ant"])

def clone():
    # clone 4 itmes
    for i in range(4):
        os.system("cp -r %s %s_%d" % (DIR_DATA_YOLO, DIR_DATA_YOLO, i))
        
if __name__ == "__main__":
    file_transfer()
    make_yolo_data()
    clone()