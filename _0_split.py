"""
This script generate a ready-to-use dataset for YOLO training

study1 and study2 will be generated

prerequisite: YOLO annotation from Roboflow. The train path must be modified by removing the relative path ".."
"""

import os
import numpy as np
from dotenv import load_dotenv
import tqdm
load_dotenv(".env")
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA_RAW  = os.getenv('DIR_DATA_RAW')
DIR_DATA_ROBO = os.getenv('DIR_DATA_ROBO')


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
        
    def filter_low_train(self, splitname):
        """
        Remove the images with low number of labels
        """
        path_lbs_train = self.ids_to_labels(self.ids[splitname])
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
        print("From %d to %d (%.2f%%)" % (len(ls_n), len(idx_keep), 100 * len(idx_keep) / len(ls_n)))
        self.ids[splitname] = list(np.array(self.ids[splitname])[idx_keep])

    
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
                # .JPEG, skip last 5 letters, otherwise, skip last 4 letters (.jpg)
                skip = -5 if prefix == 'b01' else -4
                ls_filename = [f[:skip] for f in os.listdir(os.path.join(dir_data_raw, d))]
                # loop over filenames and append it to the list
                ids[prefix] = []
                for f_raw in ls_filename:
                    # find which item in the ids["all"] contain the f_raw
                    for f_robo in ids["all"]:
                        if f_raw in f_robo:
                            ids[prefix].append(f_robo)
                            break
    # special case to avoid mixing up with t1-A1_17
    ids["b01"].append("t1-A1_1_JPEG.rf.aa31bc41fb5cd460b62715fdf93014fe")
    # return 
    return ids


def assign_new_split(ids):
    # study 1
    # new train: b02: t1 and t2, b03: t1-t5, b04: t1-t5
    ids["s1_train"] = \
        [f for f in ids["b02"] if "t1-" in f or "t2-" in f or "t3-" in f] +\
        [f for f in ids["b03"] if "t1-" in f or "t2-" in f or "t3-" in f or "t4-" in f or "t5-" in f or "t6-" in f] +\
        [f for f in ids["b04"] if "t1-" in f or "t2-" in f or "t3-" in f or "t4-" in f or "t5-" in f or "t6-" in f]
    # new test
    # similar
    ids["s1_test_a01"] = [f for f in ids["b03"] if "t7-" in f or "t8-" in f or "t9-" in f]
    ids["s1_test_a02"] = [f for f in ids["b04"] if "t7-" in f or "t8-" in f]
    ids["s1_test_a03"] = ids["b05"]
    # different
    ids["s1_test_b01"] = ids["b06"]
    ids["s1_test_b02"] = ids["a03"]
    ids["s1_test_b03"] = ids["b01"] # dense fire ant
    # pseudo test split for pyniche YOLO_API
    ids["s1_test"] = ids["b01"]
    
    # study 2
    ids["s2_train"] = ids["b02"] + ids["b03"] + ids["b04"] +\
                        ids["b05"] + ids["b06"] +\
                            ids["a01"] + ids["a02"] + ids["a03"]
    ids["s2_test"] = ids["b01"]
    
    # return
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
        subname = key[3:] # skip s1_ or s2_
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
        f.write("path: %s\n" % dir_out)
        f.write("nc: %d\n" % data.nc)
        f.write("names: ['%s']\n" % "', '".join(data.classes))
        for s in keys_new:
            key = s[3:] # skip s1_ or s2_
            f.write("%s: %s/images\n" % (key, key))

if __name__ == "__main__":
    path_yaml = os.path.join(DIR_DATA_ROBO, "data.yaml")
    data = YOLO_ROBOFLOW_API(path_yaml)
    dir_root = os.path.dirname(DIR_DATA_ROBO)
    for i in [1, 2]:
        dir_out = os.path.join(dir_root, "study%d" % i)
        data.filter_low_train("s%d_train" % i)
        keys_new = [k for k in data.ids.keys() if "s%d_" % i in k]
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