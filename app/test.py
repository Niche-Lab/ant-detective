from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import supervision as sv
import numpy as np
from PIL import Image


model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='ant_detective.pt',
    confidence_threshold=0.25,
    device="cpu", 
)

from pathlib import Path
from PIL import Image
import shutil
dir_out = Path("output")
shutil.rmtree(dir_out, ignore_errors=True)
dir_out.mkdir(exist_ok=True, parents=True)
for d in ["labels", "images", "counts"]:
    (dir_out / d).mkdir(exist_ok=True, parents=True)

ls_files = [
    "demo/img1.jpg",
    "demo/img2.jpg",
    "demo/img3.jpg",
]
ls_files = [Path(f) for f in ls_files]
ls_files[0].name


pils = [Image.open(f) for f in ls_files]
preds = predict_sahi(pils, model, 4, 0)
# preds = predict_sahi(pils, model, no_slice=True)
pils_ann = [annotate_detection(
    pil, 
    pred,
    box_color=sv.Color.RED,
    box_thickness=1,
    fill_opacity=0.15,
    fill_color=sv.Color.WHITE,
) for pil, pred in zip(pils, preds)]

for i, f in enumerate(ls_files):
    # save original images
    pils[i].save(dir_out / "images" / f.name)
    # save annotated images
    pils_ann[i].save(dir_out / "counts" / f.name)
    # save labels
    with open(dir_out / "labels" / f"{str(f.stem)}.txt", "w") as label_file:
        for det in preds[i]:
            str_det = " ".join([str(d) for d in det[0]])
            label_file.write(f"0 {str_det}\n")


det[0]

pils_ann[1]

len(preds)
preds



preds

