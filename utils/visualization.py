import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())

from pyniche.visualization.supervision import annotate_detection
import supervision as sv

def vis_preds(pils, preds, obs=None, text=True, conf=0.5):
    # handling single PIL image or list of PIL images
    if isinstance(pils, Image.Image):
        pils = [pils]
        preds = [preds]
        if obs is not None:
            obs = [obs]
    # filter low-confidence detections
    new_preds = []
    for pred in preds:
        new_preds += [pred[pred.confidence >= conf]]
    preds = new_preds

    vis_pred = []
    for det, pil in zip(preds, pils):
        if det is None:
            continue
        if text:
            out = annotate_detection(pil, det,
                [f"ant: {conf:.3f}" for conf in det.confidence], 
                box_color=sv.Color.BLUE, 
                box_thickness=1,
                text_color=sv.Color.WHITE,)
        else:
            out = annotate_detection(pil, det, 
                box_color=sv.Color.BLUE, 
                box_thickness=1, 
                text_color=sv.Color.WHITE,)
        vis_pred.append(out)
    
    if obs is not None:
        vis_obs = []
        for det, pil in zip(obs, vis_pred):
            if det is None:
                continue
            out = annotate_detection(pil, det, 
                box_color=sv.Color.RED, 
                box_thickness=1,
                text_color=sv.Color.WHITE,)
            vis_obs.append(out)
        return vis_obs
    return vis_pred

def plot_unconstrained_opt(pbounds, object_func, optimizer, obs, pils, model):
    """
    Visualizes an unconstrained 2D Bayesian optimization surface, including:
    - True objective function values
    - Surrogate model predictions
    - Acquisition function surface
    - Sampled points and best guess
    """

    # Generate grid
    divider_vals = np.arange(pbounds['divider'][0], pbounds['divider'][1] + 1)
    overlap_vals = np.linspace(pbounds['overlap'][0], pbounds['overlap'][1], 100)
    X, Y = np.meshgrid(divider_vals, overlap_vals)
    grid_points = np.array([[int(xi), yi] for xi, yi in zip(np.ravel(X), np.ravel(Y))])

    # Evaluate true objective
    Z_true = np.array([
        object_func(div, ov, obs, pils, model, no_slice=False)
        for div, ov in grid_points
    ]).reshape(X.shape)

    # Predict from GP model
    Z_pred = optimizer._gp.predict(grid_points).reshape(X.shape)

    # Compute acquisition function (assumed to be available)
    Z_acq = optimizer.acquisition(grid_points).reshape(X.shape)

    # Extract optimizer history
    res = optimizer.res
    x_ = np.array([r["params"]["divider"] for r in res])
    y_ = np.array([r["params"]["overlap"] for r in res])
    max_ = optimizer.max

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = ["True Objective", "Surrogate Prediction", "Acquisition Function"]
    data = [Z_true, Z_pred, Z_acq]

    for ax, Z, title in zip(axs, data, titles):
        contour = ax.contourf(X, Y, Z, cmap=plt.cm.coolwarm)
        ax.set_title(title)
        ax.set_xlabel("divider")
        ax.set_ylabel("overlap")
        ax.scatter(x_, y_, c='white', s=60, edgecolors='black', label="Samples")
        ax.scatter(max_["params"]["divider"], max_["params"]["overlap"], c='green', s=100, edgecolors='black', label="Max")
        ax.legend()

    return fig, axs
