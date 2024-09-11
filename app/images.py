import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os


def load_image(img, tgt_width=640):
    pil_in = Image.open(img)
    w, h = pil_in.size
    ratio = h / w
    height = int(tgt_width * ratio)
    pil_out = pil_in.resize((tgt_width, height))
    return pil_out


def caching_images(path_cache="cache"):
    if not os.path.exists(path_cache):
        os.makedirs(path_cache, exist_ok=True)
    text_bar = "Caching images..."
    bar = st.progress(0, text=text_bar)
    n_imgs = st.session_state.n_imgs
    file_imgs = st.session_state.file_imgs
    file_ram = st.session_state.file_ram
    portion = int(100 / n_imgs)
    # showing progress bar of caching images
    for i in range(n_imgs):
        filename = file_ram[i].name
        pil = load_image(file_ram[i])
        filepath = os.path.join(path_cache, filename)
        st.session_state.file_imgs[i] = filepath
        pil.save(filepath)
        bar.progress((i + 1) * portion, text=text_bar)
    bar.empty()




def avg_rgb(img_pil):
    """
    Assuming the img_pil is in mode "RGBA"
    """
    # turn 4-channel 2d array to 1d array
    rgbvec = np.array(img_pil)[:, :, :3].reshape(-1, 3)
    a_vec = np.array(img_pil)[:, :, 3].reshape(-1)
    # get average value of each channel
    return np.mean(rgbvec[a_vec > 0], axis=0)
