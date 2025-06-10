import streamlit as st

# local
from images import caching_images
from models import predict
from file_io import clean_up



def init_globals():
    ls_attr = [
        "file_ram", # list: paths (Path()) to the uploaded images (RAM)
        "file_imgs", # list: paths (str) to the cached images (local)
        "file_pred", # list: paths (str) to the annotated images
        "n_imgs",  # number of images
        "cur_i",  # current image index
        "detect_count",  # number of detections
        # system status
        "loaded", 
        "init", # bool: check if the app is initialized
        # SAHI
        "enable_sahi", # bool: whether to use SAHI
        "divider", # int: SAHI parameter
        "overlap", # float: SAHI parameter
        # visualization
        "width_lines"
    ]
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
