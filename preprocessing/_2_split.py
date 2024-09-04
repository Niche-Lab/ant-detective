"""
This script generate a ready-to-use dataset for YOLO training

prerequisite: YOLO annotation from Roboflow. The train path must be modified by removing the relative path ".."
"""

import os
import sys
import re
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import tqdm
import supervision as sv
load_dotenv('../.env')
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.getenv('DIR_DATA_ROBO')
DIR_DATA_RAW = os.getenv('DIR_DATA_RAW')
DIR_OUT = os.path.join(os.path.dirname(DIR_DATA), "final")
# sys.path.append(DIR_SRC)
# from data.yolo import YOLO_API


class YOLO_ROBOFLOW_API:
    def __init__(self, path_yaml):
        dir_yaml = os.path.dirname(path_yaml)
        with open(path_yaml) as f:
            lines_yaml = f.readlines()
            lines_yaml = [l.strip() for l in lines_yaml] # rm \n
            path_images = dict()
            path_labels = dict()
            ids = dict()
            classes = None
            nc = None
            # get classes info
            for line in lines_yaml:
                # if line is empty, skip
                if len(line) == 0:
                    continue
                if "nc" in line:
                    nc = int(line.split(":")[1].strip())
                elif "names" in line:
                    classes = line.split(":")[1].strip()
                    classes = classes.replace("[", "").replace("]", "").replace("'", "").split(",")                    
                else:
                    s = line.split(":")[0].strip()
                    if s == 'path':
                        continue
                    print("Found the split %s" % s)
                    dir_split = line.split(":")[1].strip()
                    dir_split = os.path.dirname(dir_split)
                    path_images[s] = os.path.join(dir_yaml, dir_split, "images")
                    path_labels[s] = os.path.join(dir_yaml, dir_split, "labels")
                    ls_imgs = os.listdir(path_images[s])
                    ids[s] = [os.path.splitext(f)[0] for f in ls_imgs]     
        
        # assign attributes
        self.path_yaml = path_yaml
        self.dir_yaml = dir_yaml
        self.ids = ids
        self.path_images = path_images
        self.path_labels = path_labels
        self.nc = nc
        self.classes = classes
        self.show_info()
        self.update_id()
        
    def update_id(self):
        ids = dict()
        ids["all"] = self.ids['train']
        ids = append_subset_id(ids, DIR_DATA_RAW)
        ids = assign_new_split(ids)
        self.ids = ids
    
    def __repr__(self):
        print("YOLO_ROBOFLOW_API")
        self.show_info()
        return ""
    
    def show_info(self):
        print("Available attributes:")
        print("   > path_yaml:   Absolute path to the yaml file")
        print("   > dir_yaml:    Absolute path to the dir of the yaml file")
        print("   > ids:         List of relative filenames without extension")
        print("   > path_images: Absolute path to images dir")
        print("   > path_labels: Absolute path to labels dir")
        print("   > nc:          Number of classes")
        print("   > classes:     List of class names")

    def ids_to_images(self, ids, split="train"):
        return [self.id_to_images(id, split) for id in ids]
    
    def ids_to_labels(self, ids, split="train"):
        return [self.id_to_labels(id, split) for id in ids]

    def id_to_images(self, id, split="train"):
        return os.path.join(self.path_images[split], id + ".jpg")

    def id_to_labels(self, id, split="train"):
        return os.path.join(self.path_labels[split], id + ".txt")
    
    def keys(self):
        return self.ids.keys()
        
    def filter_low_train(self):
        """
        Remove the images with low number of labels
        """
        path_lbs_train = self.ids_to_labels(self.ids['new_train'])
        # get number of lines in each label file
        ls_n = []
        n_0 = 0
        for p in path_lbs_train:
            with open(p) as f:
                n = len(f.readlines())
                if n == 0:
                    n_0 += 1
                ls_n.append(n)
        # from 1591 to 698 (43.87%)
        idx_keep = [i for i, n in enumerate(ls_n) if n > 3]
        self.ids['new_train'] = list(np.array(self.ids['new_train'])[idx_keep])

    
    
def append_subset_id(ids, dir_data_raw):
    """
    add each subset id (relative) to the ids dict
    based on the actual filenames in the raw data
    """
    ls_prefix = ["a0%d" % i for i in range(1, 4)] + ["b0%d" % i for i in range(1, 7)]
    ls_dirs = os.listdir(dir_data_raw)
    # loop over subset prefix
    for prefix in ls_prefix:
        # loop over actual dirs
        for d in ls_dirs:
            if prefix in d:
                ls_filename = [f[:-4] for f in os.listdir(os.path.join(dir_data_raw, d))]
                # loop over filenames and append it to the list
                ids[prefix] = []
                for f_raw in ls_filename:
                    # find which item in the ids["all"] contain the f_raw
                    for f_robo in ids["all"]:
                        if f_raw in f_robo:
                            ids[prefix].append(f_robo)
                            break
    # return 
    return ids

def assign_new_split(ids):
    # new train: b02: t1 and t2, b03: t1-t5, b04: t1-t5
    ids["new_train"] = [f for f in ids["b02"] if "t1-" in f or "t2-" in f] +\
        [f for f in ids["b03"] if "t1-" in f or "t2-" in f or "t3-" in f or "t4-" in f or "t5-" in f] +\
            [f for f in ids["b04"] if "t1-" in f or "t2-" in f or "t3-" in f or "t4-" in f or "t5-" in f]
    # new val: b02: t3, b03: t6, b04: t6
    ids["new_val"] = [f for f in ids["b02"] if "t3-" in f] +\
            [f for f in ids["b03"] if "t6-" in f] +\
                [f for f in ids["b04"] if "t6-" in f]
    # new test
    ids["new_test_b03"] = [f for f in ids["b03"] if "t7-" in f or "t8-" in f or "t9-" in f]
    ids["new_test_b04"] = [f for f in ids["b04"] if "t7-" in f or "t8-" in f]
    for new_test in ["b01", "b05", "b06", "a01", "a02", "a03"]:
        ids["new_test_%s" % new_test] = ids[new_test]
    return ids

def check_split_dir(dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    if not os.path.exists(os.path.join(dir_out, "images")):
        os.makedirs(os.path.join(dir_out, "images"))
    if not os.path.exists(os.path.join(dir_out, "labels")):
        os.makedirs(os.path.join(dir_out, "labels"))

def write_dataset(data, keys_new, dir_out):
    write_images_labels(data.ids, keys_new, dir_out)
    write_yaml(data, keys_new, dir_out)

def write_images_labels(ids, keys, dir_out):
    for key in keys:
        subname = key.replace("new_", "")
        dir_key = os.path.join(dir_out, subname)
        check_split_dir(dir_key)
        for id in tqdm.tqdm(ids[key], desc="Writing %s" % subname):
            # images
            img_src = data.id_to_images(id, "train")
            img_dst = os.path.join(dir_key, "images", id + ".jpg")
            os.system("cp %s %s" % (img_src, img_dst))
            # labels
            label_src = data.id_to_labels(id, "train")
            label_dst = os.path.join(dir_key, "labels", id + ".txt")
            os.system("cp %s %s" % (label_src, label_dst))

def write_yaml(data, keys_new, dir_out):
    with open(os.path.join(dir_out, "data.yaml"), "w") as f:
        f.write("path: .\n")
        f.write("nc: %d\n" % data.nc)
        f.write("names: ['%s']\n" % "', '".join(data.classes))
        for s in keys_new:
            key = s.replace("new_", "")
            f.write("%s: %s/images\n" % (key, key))


if __name__ == "__main__":
    path_yaml = os.path.join(DIR_DATA, "data.yaml")
    data = YOLO_ROBOFLOW_API(path_yaml)
    data.filter_low_train()
    keys_new = [k for k in data.ids.keys() if "new" in k]
    dir_out = os.path.join(os.path.dirname(DIR_DATA), "final")
    write_dataset(data, keys_new, dir_out)




# a01 
## ctrl:  20230404-1237 to 20230406-0837
## virus: 20230404-2336 to 20230406-1936

# a02
## ctrl:  20230324-0904 to 20230326-0834
## virus: 20151231-2004 to 20160102-1934

# a03
## ctrl: 20230329-0909 to 20230331-1209
## virus: 20230329-2008 to 20230331-2308


"""
YOLO_API

attributes
---
- path_yaml: str
    absolute path to data.yaml
- dir_yaml: str
    absolute path to the dir of data.yaml
- ls_ids: list[str]
    relative filenames without extension
- path_images and path_labels: dict
    absolute path to images/labels dir
    keys: ["train", "val", "test"]
    values: str

methods
---

"""