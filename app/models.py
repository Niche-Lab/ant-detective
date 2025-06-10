from gc import enable
import os
import shutil
import pandas as pd
import numpy as np
import time
from PIL import Image
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import supervision as sv

import streamlit as st
"""
yolo
    /labels
        t1-A1_14_4x10_8_2.txt
        t1-A1_14_4x10_8_3.txt
        ...
    /images
        t1-A1_14_4x10_8_2.jpg
        t1-A1_14_4x10_8_3.jpg
        ...
    /counts
        t1-A1_14_4x10_8_2.jpg
        t1-A1_14_4x10_8_3.jpg
        ...
"""
FILEPATH = __file__
DIR_ROOT = Path(os.path.dirname(FILEPATH))
MODEL_NAME = DIR_ROOT / "ant_detective.pt"
DIR_OUT = DIR_ROOT / "output"

model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_NAME,
    confidence_threshold=0.25,
    device="cpu", 
)

def predict():
    # init output folder
    shutil.rmtree(DIR_OUT, ignore_errors=True)
    DIR_OUT.mkdir(exist_ok=True, parents=True)
    for d in ["labels", "images", "counts"]:
        (DIR_OUT / d).mkdir(exist_ok=True, parents=True)

    # load global parameters
    ls_files = st.session_state.file_imgs
    enable_sahi = st.session_state.enable_sahi
    divider = st.session_state.divider
    overlap = st.session_state.overlap
    width_lines = st.session_state.width_lines

    print(f"Predicting {len(ls_files)} images with SAHI={enable_sahi}, divider={divider}, overlap={overlap}")
    # load images    
    pils = [Image.open(f) for f in ls_files]
    # predict
    if enable_sahi:
        preds = predict_sahi(pils, model, divider, overlap)
    else:
        preds = predict_sahi(pils, model, no_slice=True)
    # visualization
    pils_ann = [annotate_detection(
        pil, 
        pred,
        box_color=sv.Color.RED,
        box_thickness=width_lines,
        fill_opacity=0.15,
        fill_color=sv.Color.WHITE,
    ) for pil, pred in zip(pils, preds)]

    for i, f in enumerate(ls_files):
        # save original images
        pils[i].save(DIR_OUT / "images" / f.name)
        # save annotated images
        pils_ann[i].save(DIR_OUT / "counts" / f.name)
        # save labels
        with open(DIR_OUT / "labels" / f"{str(f.stem)}.txt", "w") as label_file:
            for det in preds[i]:
                str_det = " ".join([str(d) for d in det[0]])
                label_file.write(f"0 {str_det}\n")

    

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
    box_annotator = sv.BoundingBoxAnnotator(
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


# def video_frame_callback(frame):
#     nparray = frame.to_ndarray(format="bgr24")
#     img = Image.fromarray(nparray)
#     out = model(img)
#     ls_cls = out[0].names
#     ls_det_cls = out[0].boxes.cls.numpy()
#     ls_det_cls = [ls_cls[i] for i in ls_det_cls]
#     ls_det_xyxy = out[0].boxes.xyxy.numpy()
#     draw = ImageDraw.Draw(img)
#     for xyxy, name in zip(ls_det_xyxy, ls_det_cls):
#         draw.rectangle(xyxy, outline="red", width=3)
#         font = ImageFont.load_default(size=30)
#         text_x = xyxy[0]
#         text_y = xyxy[3]
#         draw.text((text_x, text_y), name, fill="red", font=font)

#     nparray = np.array(img)
#     img_out = av.VideoFrame.from_ndarray(nparray, format="bgr24")
#     return img_out



# def live_inference():
#     webrtc_streamer(key="streamer",
#                     video_frame_callback=video_frame_callback)
#     # img_file_buffer = st.camera_input("Take a picture")
#     # while img_file_buffer:
#     #     img_tmp = st.image(img_file_buffer)
#     #     # remove the img
#     #     time.sleep(1)
#     #     img_tmp.empty()     