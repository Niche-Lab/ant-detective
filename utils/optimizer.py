import os, sys
import time
import numpy as np
import pandas as pd
from PIL import Image

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from bayes_opt import BayesianOptimization, acquisition

from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())

from pyniche.data import supervision
from pyniche import evaluate

PBOUNDS = {
    'divider': (1, 4),  # factor
    'overlap': (0, 0.5),  # ratio
}
GRID_SEARCH = {
    'divider': [1, 2, 4],  # factor
    'overlap': [0, 0.25, 0.5],  # ratio
}
ACQ_FUNC = acquisition.ProbabilityOfImprovement

CONF_THRED = 0.25  # default confidence threshold for detections
CONF_OPT = 0.5  # default confidence threshold for counting-based optimization

class SAHIOptimizer:
    def __init__(self, model_path, conf_thred=CONF_THRED):
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=conf_thred,
            device="cuda:0",  # or 'cuda:0' if GPU is available
        )
        self.results = dict()
    
    def bo_optimize(self, obj_func="count", obs=None, pils=None,
                    init_points=2, n_iter=3, xi=1):
        """
        obj_func: "count" for number of high-confidence detections, 
                   "map" for mAP@0.5:0.95
        xi: exploration-exploitation trade-off parameter for acquisition function

        returns a dictionary:
        {
            "out": pd.DataFrame, target, <param1>, <param2>, ...
            "best_params": dict, best parameters found,
            "best_target": float/int, best target value found
        }
        """
        if obj_func == "count":
            func = obj_func_count
        elif obj_func == "map":
            func = obj_func_map
        
        # step 1: optimization
        time_start = time.time()
        optimizer = BayesianOptimization(
            f=lambda divider, overlap: func(
                divider, overlap, 
                obs=obs, pils=pils, 
                model=self.model),
            pbounds=PBOUNDS,
            acquisition_function=ACQ_FUNC(xi=xi),
        )
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        time_passed = np.float64(time.time() - time_start).round(2)

        # step 2: convert results to DataFrame
        ls_target = [r["target"] for r in optimizer.res]
        map_x = [r["params"] for r in optimizer.res]
        df_y = pd.Series(ls_target, name="target")
        df_x = pd.DataFrame([[k[i] for i in k.keys()] for k in map_x])
        keys = map_x[0].keys()
        df_merged = pd.concat([df_x, df_y], axis=1)
        df_merged.columns = list(keys) + ["target"]
        best_idx = df_merged["target"].idxmax()
        df_merged["divider"] = df_merged["divider"].astype(int)
        df_merged["overlap"] = df_merged["overlap"].round(2)

        # step 3: update results
        self.results["out"] = df_merged
        self.results["best_params"] = df_merged.iloc[best_idx].drop("target").to_dict()
        self.results["best_target"] = df_merged.iloc[best_idx]["target"]
        self.results["time_hptuning"] = time_passed
    
    def grid_optimize(self, obj_func="count", obs=None, pils=None):
        """
        obj_func: "count" for number of high-confidence detections, 
                   "map" for mAP@0.5:0.95
                   
        returns a dictionary:
        {
            "out": pd.DataFrame, target, <param1>, <param2>, ...
            "best_params": dict, best parameters found,
            "best_target": float/int, best target value found
        }
        """
        if obj_func == "count":
            func = obj_func_count
        elif obj_func == "map":
            func = obj_func_map
        
        # step 1: optimization
        time_start = time.time()
        ls_target = []
        ls_params = []
        for divider in GRID_SEARCH["divider"]:
            for overlap in GRID_SEARCH["overlap"]:
                target = func(divider, overlap, 
                              obs=obs, pils=pils, 
                              model=self.model,)
                ls_target.append(target)
                ls_params.append({"divider": divider, "overlap": overlap})
        time_passed = np.float64(time.time() - time_start).round(2)

        # step 2: convert results to DataFrame
        df = pd.DataFrame(ls_params)
        df["target"] = ls_target
        best_idx = df["target"].idxmax()
        df["divider"] = df["divider"].astype(int)
        df["overlap"] = df["overlap"].round(2)

        # step 3: update results
        self.results["out"] = df
        self.results["best_params"] = df.iloc[best_idx].drop("target").to_dict()
        self.results["best_target"] = df.iloc[best_idx]["target"]
        self.results["time_hptuning"] = time_passed
    
    def inference(self, pils, obs, no_slices=False):
        # step 1: predict with best parameters
        time_start = time.time()
        if no_slices:
            preds = predict_sahi(pils, self.model, no_slice=True)
            self.results["best_params"] = {"divider": 0, "overlap": 0.0}
            self.results["time_hptuning"] = 0.0
        else:
            preds = predict_sahi(pils, self.model, **self.results["best_params"])
        time_passed = np.float64(time.time() - time_start).round(2)
        self.results["time_inference"] = time_passed
        self.results["predictions"] = preds

        # step 2: evaluate predictions
        metrics = evaluate.from_sv(preds, obs)

        return dict({
            "time_hptuning": self.results["time_hptuning"],
            "time_inference": self.results["time_inference"],
            "divider": self.results["best_params"]["divider"],
            "overlap": self.results["best_params"]["overlap"],
            **metrics,
        })
    
    
def obj_func_map(
    divider, overlap, # search parameters
    obs, pils, # ground truth and input images
    model, # model to use for prediction 
    no_slice=False, # if True, no slicing is applied, otherwise the image is sliced
    **kwargs
):
    preds = predict_sahi(pils, model, divider, overlap, no_slice=no_slice)
    return evaluate.from_sv(preds, obs)["map5095"]

def obj_func_count(
    divider, overlap, # search parameters 
    pils, # input images 
    model, # model to use for prediction 
    no_slice=False, # if True, no slicing is applied, otherwise the image is sliced
    **kwargs # additional arguments (not used)
):
    preds = predict_sahi(pils, model, divider, overlap, no_slice=no_slice)
    return count_det(preds)

def count_det(preds, conf_thred=CONF_OPT):
    preds = handle_single(preds)
    count = 0
    for pred in preds:
        count += (pred.confidence > conf_thred).sum()
    return int(count)


def predict_sahi(pils, model, divider=1, overlap=0, no_slice=False, swsh=None):
    pils = handle_single(pils)

    divider = int(divider)
    overlap = float(overlap)

    preds = []
    for pil in pils:
        imgW, imgH = pil.size
        if no_slice:
            sh = imgH
            sw = imgW
        elif swsh is not None:
            sw = swsh[0]
            sh = swsh[1]
        else:
            short_side = imgW if imgW < imgH else imgH
            sh = short_side // divider
            sw = short_side // divider
        
        result = get_sliced_prediction(
            pil, model,
            verbose=0,
            slice_height=sh,
            slice_width=sw,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
        )
        pred = supervision.from_sahi_to_dets(result)
        preds.append(pred)
    if len(preds) == 1:
        return preds[0]
    return preds  # return list of predictions if multiple images are provided

def handle_single(obj):
    """if the obje is a single object, convert it to a list"""
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


# # example --------------------------
# def black_box_function(x, y, z):
#     """Function with unknown internals we wish to maximize.

#     This is just serving as an example, for all intents and
#     purposes think of the internals of this function, i.e.: the process
#     which generates its output values, as unknown.
#     """
#     return -x ** 2 - (y - 1) ** 2 + 1 + z ** 2

# optimizer = BayesianOptimizer(
#     func=lambda x, y: black_box_function(x, y, 0.5),
#     search_bounds={'x': (2, 4), 'y': (-3, 3)},
#     random_state=1,
# )
# optimizer.maximize()
# optimizer.max()
# optimizer.out()
