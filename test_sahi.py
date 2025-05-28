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

LS_TEST = ["test_a01", "test_a02", "test_a03", 
           "test_b01", "test_b02", "test_b03"]

yolo = YOLO_API("/home/niche/data/finding-ants/study1/")
test_split = yolo["test_a01"]

import random
idx_rdm = random.sample(range(len(test_split)), 10)

obs = test_split.get_detections(idx_rdm)
pils = test_split.get_PILs(idx_rdm)


optimizer = SAHIOptimizer(model_path="out/best.pt")
results = dict({})

results["bo_eplor"] = optimizer.bo_optimize(
    obj_func="count", pils=pils,
    init_points=3, n_iter=3, xi=10,)
results["bo_eploi"] = optimizer.bo_optimize(
    obj_func="count", pils=pils,
    init_points=3, n_iter=3, xi=2,)
results["grid_count"] = optimizer.grid_optimize(
    obj_func="count", pils=pils,)

preds = predict_sahi(pils=pils, model=optimizer.model, no_slice=True) 
evaluate.from_sv(preds, obs)

preds = predict_sahi(pils=pils, model=optimizer.model,
                     **results["bo_eplor"]["best_params"]) 
evaluate.from_sv(preds, obs)

preds = predict_sahi(pils=pils, model=optimizer.model,
                     **results["bo_eploi"]["best_params"]) 
evaluate.from_sv(preds, obs)

preds = predict_sahi(pils=pils, model=optimizer.model,
                     **results["grid_count"]["best_params"]) 
evaluate.from_sv(preds, obs)




preds = predict_sahi(pils=pils, model=optimizer.model,
                     **results["bo_map"]["best_params"])
evaluate.from_sv(preds, obs)["precision"].round(3)

results["bo_map"] = optimizer.bo_optimize(
    obj_func="map", obs=obs, pils=pils,
    init_points=3, n_iter=3,
    strategy="exploration",)

results["out"]


preds = []
for pil in pils:
    pred = predict_sahi(
        pil=pil, model=optimizer.model,
        divider=1, overlap=0, no_slice=False)
    preds.append(pred)    
evaluate.from_sv(preds, obs)["map5095"]


preds = []
for pil in pils:
    pred = predict_sahi(
        pil=pil, model=optimizer.model,
        divider=3, overlap=0.24689, no_slice=False)
    preds.append(pred)    
evaluate.from_sv(preds, obs)["map5095"]


    
count_det(preds, conf_thred=0.6)


preds
viss = vis_preds(preds, obs, pils)
viss[1]



divider = 2
overlap = 0.25
no_slice=False
def object_func(divider, overlap, obs, pils, model, no_slice=False):
divider = int(divider)
overlap = float(overlap)
preds = []
for pil in pils:
    imgW, imgH = pil.size
    result = get_sliced_prediction(
        pil, model,
        verbose=0,
        slice_height = imgH if no_slice else imgH // divider,
        slice_width = imgW if no_slice else imgH // divider,
        overlap_height_ratio = overlap,
        overlap_width_ratio = overlap,)
    preds += [supervision.from_sahi_to_dets(result)]


np_pils = np.array([np.array(pil) for pil in pils])
np_pils.shape

result = get_sliced_prediction(
        np_pils, model,
        verbose=0,
        slice_height = imgH if no_slice else imgH // divider,
        slice_width = imgW if no_slice else imgH // divider,
        overlap_height_ratio = overlap,
        overlap_width_ratio = overlap,)

def count_det(preds, conf_thred=0.6):
    count = 0
    for pred in preds:
        count += (pred.confidence > conf_thred).sum()
    return count
count_det(preds)

THRED = 0.6
pred = preds[0]
(pred.confidence > THRED).sum()
len(pred) 


return from_sv(preds, obs)["map5095"]


pbounds = {
        'divider': (1, 8),  # factor
        'overlap': (0, 0.5),  # ratio
    }
optimizer = BayesianOptimizer(
    func=lambda divider, overlap: object_func(
        divider, overlap, obs, pils, model, False),
    search_bounds=pbounds,
    xi=0.1,
)
optimizer.maximize(init_points=3, n_iter=7)

object_func(0, 0, obs, pils, model, no_slice=True)
object_func(1, 0.5, obs, pils, model, no_slice=False)

optimizer.optimizer._gp.predict
optimizer.
optimizer.max()
{'target': 0.38252428571428576,
 'params': {'divider': 1.054653436503521, 'overlap': 0.09336952296537931}}

{'target': 0.38252428571428576,
 'params': {'divider': 1.0761225494773352, 'overlap': 0.48833565145117624}}

