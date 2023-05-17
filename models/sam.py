import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor
import os

DIR_MODEL = os.path.abspath(os.path.dirname(__file__))
SAM_CHECKPOINT_PATH = os.path.join(DIR_MODEL, "sam_vit_h_4b8939.pth")
SAM_ENCODER_VERSION = "vit_h"


def init_sam():
    "set model architecture and load weights"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return SamPredictor(sam)


def seg_by_sam(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    """
    segment image with sam_predictor
    ---
    params:
        sam_predictor: SamPredictor
        image: np.ndarray of shape (h, w, c) with c=3
        xyxy: np.ndarray of shape (n, 4) with n bounding boxes
    return:
        np.ndarray of shape (n, h, w) with n binary masks
        w and h are the width and height of the image
        can be set as detection.mask
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box[None, :], multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def vis_detections(image: np.ndarray, detection: sv.Detections, figsize=(8, 8)) -> None:
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detection)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detection
    )
    sv.plot_image(annotated_image, figsize)
