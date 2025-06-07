import sys
from paths import PathFinder
from sahi import AutoDetectionModel

from ultralytics import YOLO
from utils.optimizer import predict_sahi

PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API
from pyniche.visualization.supervision import annotate_detection


model_a = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", 
    model_path="out/study2/thread_0/yolo11n_1_NOFT/iter_98/weights/best.pt",
    confidence_threshold=0.3,)
model_bc = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", 
    model_path="out/study2/thread_0/yolo11n_2_FT/iter_98/weights/best.pt",
    confidence_threshold=0.3,)
data = YOLO_API(PATHS["DIR_DATA"] / "study2")
data_dense = data["test_dense"]
data_inat = data["test_inat"]


import supervision as sv
param_vis = {
    "box_color": sv.Color.BLUE,
    "box_thickness": 2,
}

pil_dense = data_dense.get_PIL(5)

pred_base = predict_sahi(
    pil_dense,
    model_a,
    no_slice=True,
)
img_base = annotate_detection(
    pil_dense,
    pred_base,
    **param_vis
)
img_base.save("out/s2_base_dense.jpg")

pred_good = predict_sahi(
        pil_dense,
        model_bc,
        divider=4, overlap=0,
    )
img_good = annotate_detection(
    pil_dense,
    pred_good,
    **param_vis
)
img_good.save("out/s2_good_dense.jpg")

pil = data_inat.get_PIL(4)
pred = predict_sahi(pil, model_a, divider=2, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_4.jpg")

pil = data_inat.get_PIL(7)
pred = predict_sahi(pil, model_bc, divider=2, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_7.jpg")

pil = data_inat.get_PIL(15)
pred = predict_sahi(pil, model_bc, divider=3, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_15.jpg")
    
pil = data_inat.get_PIL(17)
pred = predict_sahi(pil, model_bc, divider=3, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_17.jpg")
    
pil = data_inat.get_PIL(18)
pred = predict_sahi(pil, model_bc, divider=3, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_18.jpg")

pil = data_inat.get_PIL(19)
pred = predict_sahi(pil, model_bc, divider=3, overlap=0)
annotate_detection(pil, pred, **param_vis).\
    save("out/s2_inat_19.jpg")

    