from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import supervision as sv
import numpy as np
from PIL import Image


def predict_sahi(pils, model, divider=1, overlap=0, no_slice=False, swsh=None):
    pils = handle_single(pils)

    divider = round(divider)
    if overlap >= -0.125 and overlap < 0.125:
        overlap = 0.0
    elif overlap >= 0.125 and overlap < 0.375:
        overlap = 0.25
    elif overlap >= 0.375 and overlap < 0.625:
        overlap = 0.5

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
        pred = from_sahi_to_dets(result)
        preds.append(pred)
    if len(preds) == 1:
        return preds[0]
    return preds  # return list of predictions if multiple images are provided


def from_sahi_to_dets(sahi):
    """
    Convert a SAHI object to a supervision.Detections.
    
    Args:
        sahi: A SAHI object from sahi.predict.get_sliced_prediction
        
    Returns:
        supervision.Detections: A supervision Detections object.

    """

    out = sahi.to_coco_annotations()
    if len(out) == 0:
        return sv.Detections.empty()
    else:
        xywh = [o["bbox"] for o in out]
        xyxy = [[x, y, x + w, y + h] for x, y, w, h in xywh]
        conf = [o["score"] for o in out]
        class_id = [o["category_id"] for o in out]
        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32).round(3),
            confidence=np.array(conf, dtype=np.float32).round(3),
            class_id=np.array(class_id, dtype=np.int64)
        )


def annotate_detection(
    image, # PIL.Image or np.ndarray
    detections, # sv.Detections containing multiple detections
    labels=None, # single label or list of labels
    # box parameters
    box_color=sv.Color.BLUE,
    box_thickness=2,
    fill_opacity=0.15,
    fill_color=sv.Color.WHITE,
    # text parameters
    text_scale=0.7,
    text_thickness=1,
    text_padding=6,
    text_color=sv.Color.BLACK,
    # other parameters
    *args,
    **kwargs,
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)


    # init annotators
    box_annotator = sv.BoxAnnotator(
        color=box_color,
        thickness=box_thickness,
    )
    fill_annotator = sv.ColorAnnotator(
        color=fill_color,
        opacity=fill_opacity,
    )
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
    )
    annotated_frame = fill_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels] * len(detections)
        label_annotator = sv.LabelAnnotator(
            color=box_color,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            smart_position=True,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
    return annotated_frame


def handle_single(obj):
    """if the obje is a single object, convert it to a list"""
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='ant_detective.pt',
    confidence_threshold=0.25,
    device="cpu", 
)

pils = [
    Image.open("demo/img1.jpg"),
    Image.open("demo/img2.jpg"),
    Image.open("demo/img3.jpg"),]
preds = predict_sahi(pils, model, 2, 0)
preds


preds

