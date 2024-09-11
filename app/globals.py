import streamlit as st

# local
from images import caching_images

# attributes
"""
file_ram: list of file paths on RAM from uploader
file_imgs: list of file paths
file_pred: list of file paths
n_imgs: number of images
cur_i: current image index

"""

def init_globals():
    # Images -----------------------------------------
    if "pil_imgs" not in st.session_state:
        st.session_state.pil_imgs = []

    # VARIABLES -----------------------------------------
    if "cur_i" not in st.session_state:
        st.session_state.cur_i = 0


def update_globals():
    n_imgs = st.session_state.n_imgs
    print("n_imgs:", n_imgs)
    st.session_state.file_imgs = [None for _ in range(n_imgs)]
    st.session_state.file_pred = [None for _ in range(n_imgs)]
    st.session_state.cur_i = 0    
    caching_images()
