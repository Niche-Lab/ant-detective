import supervision as sv
import numpy as np

# local imports
from models.sam import seg_by_sam
from libs import compute_area


class Niche_Batch:
    def __init__(self):
        pass


class Niche_CV_Object(sv.Detections):
    def __init__(self, image, xyxy, thred=1e9):
        xyxy = preprocess_xyxy(xyxy, thred)
        super().__init__(xyxy)
        self.image = image
        self.mask = None

    def set_mask_by_sam(self, sam_predictor):
        self.mask = seg_by_sam(sam_predictor, self.image, self.xyxy)

    def plot_image(self, figsize=(8, 8)):
        annt_img = self.get_annotated_img()
        sv.plot_image(annt_img, figsize)

    def get_annotated_img(self):
        annotated_image = self.image.copy()
        # annotate mask
        if self.mask is not None:
            annotator = sv.MaskAnnotator()
            annotated_image = annotator.annotate(annotated_image, self)
        # annotate box
        annotator = sv.BoxAnnotator()
        annotated_image = annotator.annotate(annotated_image, self)
        return annotated_image

    def get_

def preprocess_xyxy(xyxy, thred=1e9):
    xyxy = np.array(xyxy, dtype=int).reshape(-1, 4)
    # filter by bbox area
    areas = []
    for box in xyxy:
        areas.append(compute_area(box))
    xyxy = xyxy[np.array(areas) < thred]
    return xyxy
