import streamlit as st
import datetime

# local imports
from callbacks import slide_seg


def show_sidebar():
    with st.sidebar:
        # if loaded:
        #     show_cropped_images()
        #     show_annotations()
        show_about()


def show_threshold_slider():
    st.subheader("Thresholding Strength")
    st.slider(
        "Thresholding Strength",
        min_value=0,
        max_value=10,
        value=0,
        on_change=slide_seg,
        key="slider_seg",
        label_visibility="collapsed",
    )


def show_cropped_images():
    st.header("Cropped Images")
    show_threshold_slider()
    cur_i = st.session_state.cur_i
    imgs = st.session_state.cropped_imgs[cur_i]
    if imgs:
        for img in imgs:
            st.image(img, use_column_width=True)


def show_annotations():
    st.header("Annotations")
    st.json(st.session_state.json_out)


def show_about():
    st.divider()
    st.subheader("James Chen (<niche@vt.edu>)")
    st.subheader("Scotty Yang (<scottyyang@vt.edu>)")
    st.write(
        """
        School of Animal Sciences, Virginia Tech

        Blacksburg, VA, USA
        """
    )
    st.divider()
    today = datetime.date.today().strftime("%B %d, %Y")
    st.write("Last updated:", today)
