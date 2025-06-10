from PIL import Image
import pandas as pd
import streamlit as st
import time
import os

# local imports
from callbacks import next_img, prev_img
from file_io import inspect_results
from globals import update_globals
from models import predict



def show_download(name_zip="ant-detective.zip"):
    if st.session_state['loaded'] and os.path.exists(name_zip):
        with open(name_zip, 'rb') as f:
            bytes = f.read()
        st.download_button("Download the Detection Results", 
                            data=bytes, 
                            type="primary",
                            file_name=name_zip, )    

def module_sahi():
    if st.session_state.enable_sahi is None:
        st.session_state.enable_sahi = True
    if st.session_state.divider is None:
        st.session_state.divider = 2
    if st.session_state.overlap is None:
        st.session_state.overlap = 0
    
    def on_sahi_change():
        # Call as many functions as you want here
        setattr(st.session_state, 'enable_sahi', st.session_state.checkbox_sahi)
        setattr(st.session_state, 'divider', st.session_state.slider_divider)
        setattr(st.session_state, 'overlap', st.session_state.slider_overlap)
        if st.session_state.init is not None:
            predict()

    
    st.markdown("### SAHI Parameters")
    st.markdown(
        "[SAHI (Slicing Aided Hyper Inference)](https://github.com/obss/sahi) is a technique to improve the detection accuracy by slicing the image into smaller parts. "
        "You can adjust the parameters below to control the slicing behavior. "
    )
    st.markdown(
        "`Slice Divider` is the number of slices in each dimension (e.g., 2 means 2x2 slices, 4 means 4x4 slices). "
    )
    st.markdown(
        "`Slice Overlap` is the percentage of overlap between the slices. "
        "For example, a value of 0.25 means that each slice will overlap with the next slice by 25% of its width/height."
    )
    st.divider()
    st.checkbox(
        "Enable SAHI",
        value=st.session_state.enable_sahi,
        on_change=on_sahi_change,
        key="checkbox_sahi",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Slice Divider",
            min_value=1,
            max_value=4,
            step=1,
            value=st.session_state.divider,
            on_change=on_sahi_change,
            key="slider_divider",
        )
    with col2:
        st.number_input(
            "Slice Overlap",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.overlap),
            step=0.25,
            on_change=on_sahi_change,
            key="slider_overlap",
        )
    st.divider()

    
def show_width_lines():
    def on_width_lines_change():
        # Call as many functions as you want here
        setattr(st.session_state, 'width_lines', st.session_state.slider_width_lines)
        predict()
    
    if st.session_state.width_lines is None:
        st.session_state.width_lines = 1
    st.markdown("### Line Width for Bounding Boxes")
    st.number_input(
        "",
        min_value=1,
        max_value=3,
        value=st.session_state.width_lines,
        step=1,
        on_change=on_width_lines_change,
        key="slider_width_lines",
        label_visibility="collapsed",
    )
    st.divider()

def show_navigator():
    cur_i = st.session_state.cur_i
    n_imgs = st.session_state.n_imgs

    if n_imgs == 0 or n_imgs == 1:
        st.empty()
    else:
        col_b1, col_b2 = st.columns([3, 1])
        col_b1.button(
            "⬅︎ Previous Image",
            on_click=prev_img,
        )
        col_b2.button(
            "Next Image ➡︎",
            on_click=next_img,
        )
        # st.success("Drag the slider to navigate between images")
        st.slider(
            "File Index",
            min_value=0,
            max_value=n_imgs - 1,
            value=cur_i,
            on_change=slide_i,
            key="slider_index",
            label_visibility="collapsed",
        )



def image_uploader():
    file_ram = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )
    is_change = file_ram != st.session_state.file_ram
    st.session_state.file_ram = file_ram
    if st.session_state.init is not None and is_change:
        print("new update!")
        update_globals() # trigger model prediction
        st.session_state.detect_count = inspect_results()

    #to avoid file_uplaoder to trigger update_globals the first time 
    st.session_state.init = True
    

def slide_i():
    slider_value = st.session_state.slider_index
    print("callback: slide_i (%d, %d)" % (slider_value, st.session_state.cur_i))
    if slider_value != st.session_state.cur_i:
        print("change index!")
        st.session_state.cur_i = slider_value
        print("after changed:", slider_value)



# class Timer:
#     def __init__(self, message="Page loaded"):
#         self.message = message

#     def __enter__(self):
#         self.start = time.time()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         end = time.time()
#         st.info(f"{self.message}: {end - self.start:.2f} seconds")



# def show_ui():
#     st.divider()
#     tg1, tg2 = st.columns(2)
#     with tg1:
#         tog_edit = st.toggle("Transform the bounding boxes", False, key="toggle_edit")
#     with tg2:
#         tog_auto = st.toggle("Render on-the-fly (slower)", True, key="toggle_auto")

#     if st.session_state.toggle_edit:
#         st.success("Drag the corners to transform the bounding boxes")
#     else:
#         st.success("Draw a rectangle on the canvas to create a new bounding box")
#     if not st.session_state.toggle_auto:
#         st.success("Right-click on the canvas to render the annotations")

#     return tog_auto, tog_edit