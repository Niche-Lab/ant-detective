import os
import sys
from PIL import Image

from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())

from pyniche.data.yolo.API import YOLO_API
from pyniche import evaluate
from utils.optimizer import SAHIOptimizer, predict_sahi
from utils.visualization import vis_preds

import random
LS_TEST = ["test_a01", "test_a02", "test_a03", 
           "test_b01", "test_b02", "test_b03"]

yolo = YOLO_API("/home/niche/data/finding-ants/study2/")
n_dense = len(yolo["test_dense"])
idx_train = random.sample(range(n_dense), 1)
idx_test = set(range(n_dense)) - set(idx_train)
thread = "3"

n_slices = [(4, 10), (4, 5), (2, 5), (4, 4), (2, 4), (4, 2), (2, 2)]
yolo.slice_split(
    split_src="test_dense", 
    split_dst="train_dense", 
    n_slices=n_slices,
    ls_idx=idx_train,)
yolo.shuffle_train_val(
    split_src=f"train_dense",
    k=10,
    suffix=f"{thread}")
yolo.make_split(
    split_src="test_dense", 
    split_dst="test", 
    ls_idx=idx_test,
    suffix=f"{thread}")
yolo.save_yaml(classes=["ant"], suffix=thread)


test_split = yolo["test_dense"]
test_split[3]


yolo.root/"val_"
idx = 3
# parameters
n_slice_x = 4
n_slice_y = 3
split_dst = f"test_{n_slice_x}x{n_slice_y}"
import shutil

# check directory
dir_src = test_split.root
dir_dst = dir_src.parent / split_dst
if os.path.exists(dir_dst):
    shutil.rmtree(dir_dst)
os.makedirs(dir_dst, exist_ok=True)
os.makedirs(dir_dst / "images", exist_ok=True)
os.makedirs(dir_dst / "labels", exist_ok=True)

# load info
path_img = test_split.images[idx]
path_lb = test_split.labels[idx]
img = test_split.get_PIL(idx)
width, height = img.size

# calculate patch sizes and handle last patch sizes
width_patch = width // n_slice_x
height_patch = height // n_slice_y
width_patch_last = width - width_patch * (n_slice_x - 1)
height_patch_last = height - height_patch * (n_slice_y - 1)
patchs_x = [width_patch] * (n_slice_x - 1) + [width_patch_last]
patchs_y = [height_patch] * (n_slice_y - 1) + [height_patch_last]

for i, patch_x in enumerate(patchs_x):
    for j, patch_y in enumerate(patchs_y):
        # define filename
        name_img_patch = f"{path_img.stem}_{n_slice_x}x{n_slice_y}_{i}_{j}.jpg"
        name_lb_patch  = f"{path_lb.stem}_{n_slice_x}x{n_slice_y}_{i}_{j}.txt"
        
        # image ------------------------------
        # calculate coordinates
        x_start = sum(patchs_x[:i])
        y_start = sum(patchs_y[:j])
        x_end = x_start + patch_x
        y_end = y_start + patch_y
        # slice the image
        img_patch = img.crop((x_start, y_start, x_end, y_end))

        # label ------------------------------
        labels_patch = []
        with open(path_lb, 'r') as f:
            labels = f.readlines()
        for label in labels:
            class_id, cx, cy, w, h = map(float, label.split())
            # denormalize coordinates
            abs_cx = cx * width
            abs_cy = cy * height
            abs_w = w * width
            abs_h = h * height
            
            # check if the bounding box is in the current patch
            if (x_start <= abs_cx <= x_end) and (y_start <= abs_cy <= y_end):
                # adjust the bounding box relative to the new patch
                new_cx = (abs_cx - x_start) / patch_x
                new_cy = (abs_cy - y_start) / patch_y
                new_w = abs_w / patch_x
                new_h = abs_h / patch_y
                labels_patch.append(f"{int(class_id)} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
        # save
        img_patch.save(dir_dst / "images" / name_img_patch)
        with open(dir_dst / "labels" / name_lb_patch, 'w') as f:
            f.writelines(labels_patch)
        


split_patch = yolo[split_dst]
split_patch[11]
    
    
    
test_split.check_keypoints()

test_split = yolo["test_inat"]

import random

obs = test_split.get_detections()
pils = test_split.get_PILs()

pil_test = pils[i]
obs_test = obs[i]

optimizer = SAHIOptimizer(model_path="out/study2.pt")
optimizer.bo_optimize(obj_func="count", pils=pil_test, xi=1)
pred_test = predict_sahi(pils=pil_test, model=optimizer.model,
                     **optimizer.results["best_params"])
pred_test = predict_sahi(pils=pil_test, model=optimizer.model, no_slice=True,)
vis_preds(pil_test, pred_test, obs_test, text=False, conf=0.5)[0]
evaluate.from_sv(pred_test, obs_test)
# good: 4, 7, 15, 17
# large: 18
# red: 19



for pred in [pred_test]:
    print(pred)



optimizer_count = SAHIOptimizer(model_path="out/study2.pt")
optimizer_count.bo_optimize(obj_func="count", pils=pils, xi=500)
optimizer_map = SAHIOptimizer(model_path="out/study2.pt")
optimizer_map.bo_optimize(obj_func="map", obs=obs, pils=pils, xi=0.1)

optimizer_count.results["best_params"]
optimizer_map.results["best_params"]

preds = predict_sahi(pils=pils, model=optimizer_map.model,
                     **optimizer_map.results["best_params"])
evaluate.from_sv(preds, obs)
viss = vis_preds(preds, obs, pils, text=False)
viss[2]



preds_count = predict_sahi(pils=pils, model=optimizer_count.model,
                     **optimizer_count.results["best_params"])
evaluate.from_sv(preds_count, obs)

i = 2
pil_test = pils[i]
obs_test = obs[i]





pred_test = predict_sahi(pils=pil_test, model=optimizer_count.model,
                     swsh=(409, 409))
from_sv([pred_test], [obs_test])

vis_preds(pil_test, pred_test, obs_test, text=False)[0]

pred_test_hc = pred_test[pred_test.confidence > 0.4]
vis_preds(pil_test, pred_test_hc, text=False)[0]


pred_test_hc = pred_test[pred_test.confidence > 0.5]
vis_preds(pil_test, pred_test_hc, obs_test, text=False)[0]



viss_ct = 
viss_ct[0]




viss_ct[2]


confs = [d]

pil_test.size

1636 2180

1636//4

2180//409

4 x 5