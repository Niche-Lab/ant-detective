import streamlit as st
import os
import shutil
# local imports
from yolo import predict, live_inference
from callbacks import enable_hotkeys
# from outputs import show_ann_count, show_output_df
from sidebar import show_sidebar
from widgets import (
    image_uploader,
    show_navigator,
    show_download,
)
import datetime
from globals import init_globals
from file_io import inspect_results, clean_up

st.set_page_config(
    page_title="Ant Detective",
    page_icon="🐜",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
enable_hotkeys()

def main():
    st.title("Ant Detective 🔎 🐜")
    init_globals()  
    image_uploader()
    
    show_download()        
    if st.session_state["loaded"]:
        detect_count = st.session_state["detect_count"]
        img = st.empty()        
        if detect_count == 0:
            st.error("No ants detected")
        else:
            show_navigator()
            cur_i = st.session_state["cur_i"]
            img.image(st.session_state["file_pred"][cur_i])   
    else:      
        # initial message
        st.success("Please upload images to get started")
        st.write("or live stream from your webcam")
        live_inference()
        # st.button("Live Inference", 
        #         key="live-button", 
        #         type="primary",
        #         on_click=live_inference)
                 
    # left-hand side
    show_sidebar()


main()
