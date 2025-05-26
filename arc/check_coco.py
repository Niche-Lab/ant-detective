from paths import PathFinder
from pathlib import Path
import sys
PATHS = PathFinder()
# insert to the first position
sys.path.insert(0, str(PATHS["DIR_PYNICHE"]))

from pyniche.data.coco.API import COCO_API


path_json_train = PATHS["DIR_iNATURALIST"] / "train_2017_bboxes.json"
coco_train = COCO_API(path_json_train)

coco_train.get_ann_by_image_id([430720])

cat_names = [
  	"Pogonomyrmex barbatus",
	"Linepithema humile",
	"Solenopsis invicta",
	"Pseudomyrmex gracilis",

    "Atta texana",
    "Camponotus vagus",
    "Camponotus pennsylvanicus",
    "Liometopum occidentale",
    "Oecophylla smaragdina",
    "Prenolepis imparis",
]

cat_ids = [c["id"] for c in coco_train.categories() if c["name"] in cat_names]
ann_matched = [a for a in coco_train.annotations() if a["category_id"] in cat_ids]
img_ids_matched = [a["image_id"] for a in ann_matched]
# count the number of each image
import pandas as pd
ids_matched = pd.Series(img_ids_matched).value_counts()
# count greater than 5
ids_matched = ids_matched[ids_matched > 3]
coco_train_sub = coco_train.subset_by_image_ids(ids_matched.index.tolist())
pil = coco_train_sub.get_img_by_idx(5)
pil = coco_train_sub.get_PIL(5)
ls_bbox = [a["bbox"] for a in coco_train_sub.get_ann_by_image_id([430720])]
# draw bbox in pil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig, ax = plt.subplots(1)
ax.imshow(pil)
for bbox in ls_bbox:
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()



# get the images


from PIL import Image
Image.open(filenames[3])


coco_train.subset_by_dir(path_json_insecta)


coco_train.get_PIL(3)



coco_train.get_(3)