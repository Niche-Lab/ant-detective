from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import os
import shutil
import pandas as pd
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
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
MODEL_NAME = os.path.join(os.path.dirname(FILEPATH), "ant_detective.pt")


model = YOLO(MODEL_NAME)
# model = YOLO("yolov8n.pt")

def video_frame_callback(frame):
    nparray = frame.to_ndarray(format="bgr24")
    img = Image.fromarray(nparray)
    out = model(img)
    ls_cls = out[0].names
    ls_det_cls = out[0].boxes.cls.numpy()
    ls_det_cls = [ls_cls[i] for i in ls_det_cls]
    ls_det_xyxy = out[0].boxes.xyxy.numpy()
    draw = ImageDraw.Draw(img)
    for xyxy, name in zip(ls_det_xyxy, ls_det_cls):
        draw.rectangle(xyxy, outline="red", width=3)
        font = ImageFont.load_default(size=30)
        text_x = xyxy[0]
        text_y = xyxy[3]
        draw.text((text_x, text_y), name, fill="red", font=font)

    nparray = np.array(img)
    img_out = av.VideoFrame.from_ndarray(nparray, format="bgr24")
    return img_out


def live_inference():
    webrtc_streamer(key="streamer",
                    video_frame_callback=video_frame_callback)
    # img_file_buffer = st.camera_input("Take a picture")
    # while img_file_buffer:
    #     img_tmp = st.image(img_file_buffer)
    #     # remove the img
    #     time.sleep(1)
    #     img_tmp.empty()        
        
        
def predict():
    ls_img = st.session_state.file_imgs
    dir_img = os.path.dirname(ls_img[0])
    model.predict(
        dir_img, 
        save=True,
        show_conf=False,
        line_width=1,
        show_labels=False,
        save_txt=True,
        project="yolo",
        name=".",
    )
  
