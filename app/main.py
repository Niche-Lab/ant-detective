import streamlit as st
import os
import shutil
# local imports
from globals import init_globals, update_globals
from yolo import predict, create_download_zip
# from callbacks import enable_hotkeys
# from outputs import show_ann_count, show_output_df
from sidebar import show_sidebar
from widgets import (
    image_uploader,
#     show_navigator,
#     show_ui,
)
import datetime


st.set_page_config(
    page_title="Ant Detective",
    page_icon="🐜",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
# enable_hotkeys()
st.title("Ant Detective 🐜")
st.session_state.file_ram = image_uploader()
st.session_state.n_imgs = len(st.session_state.file_ram)
loaded = st.session_state.n_imgs > 0
if not loaded:
    st.success("Please upload images to get started")
else:
    clean_up()
    
    update_globals()
    predict()
    st.image(st.session_state.file_pred[0])
    st.success("All images are loaded successfully")
    # yyyymmdd-hhmm
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    create_download_zip("yolo", f"{today}")
 

# left-hand side
show_sidebar()


def clean_up():
    shutil.rmtree("yolo", ignore_errors=True)
    shutil.rmtree("cache", ignore_errors=True)
    for f in os.listdir():
        if f.endswith(".zip"):
            os.remove(f)