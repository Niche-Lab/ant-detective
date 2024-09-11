import time
import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts


def change_cur_i(i):
    # only save json_data when switching images
    cur_i = st.session_state.cur_i
    st.session_state.json_data[cur_i] = st.session_state.json_data_tmp
    st.session_state.cur_i = i
    print("after changed:", i)


def slide_seg():
    slider_value = st.session_state.slider_seg
    print("callback: slide_seg (%d, %d)" % (slider_value, st.session_state.seg_binary))
    if slider_value != st.session_state.seg_binary:
        print("change strength!")
        st.session_state.seg_binary = slider_value


def slide_i():
    slider_value = st.session_state.slider_index
    print("callback: slide_i (%d, %d)" % (slider_value, st.session_state.cur_i))
    if slider_value != st.session_state.cur_i:
        print("change index!")
        change_cur_i(slider_value)


def next_img():
    print("callback: next_img")
    i = st.session_state.cur_i
    n_imgs = st.session_state.n_imgs
    i += 1
    if i >= n_imgs:
        i = 0
    change_cur_i(i)


def prev_img():
    print("callback: prev_img")
    i = st.session_state.cur_i
    n_imgs = st.session_state.n_imgs
    i -= 1
    if i < 0:
        i = n_imgs - 1
    change_cur_i(i)


def enable_hotkeys():
    add_keyboard_shortcuts(
        {
            "ArrowRight": "Next Image",  # "button_next
            "ArrowLeft": "Previous Image",
        }
    )
