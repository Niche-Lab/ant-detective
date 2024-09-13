import streamlit as st

# local
from images import caching_images
from yolo import predict
from file_io import clean_up

# attributes
"""
file_ram: list of file paths on RAM from uploader
file_imgs: list of file paths
file_pred: list of file paths
n_imgs: number of images
cur_i: current image index
init: to avoid file_uplaoder to trigger update_globals the first time 
"""

def init_globals():
    ls_attr = ["file_ram", "file_imgs", "file_pred", 
               "n_imgs", "cur_i", "detect_count",
               "loaded", "init"]
    for attr in ls_attr:
        if attr not in st.session_state:
            st.session_state[attr] = None

def update_globals():
    clean_up()
    n_imgs = len(st.session_state.file_ram)
    st.session_state.loaded = True
    st.session_state.n_imgs = n_imgs
    print("n_imgs:", n_imgs)
    st.session_state.file_imgs = [None for _ in range(n_imgs)]
    st.session_state.file_pred = [None for _ in range(n_imgs)]
    st.session_state.cur_i = 0    
    caching_images()
    predict()
