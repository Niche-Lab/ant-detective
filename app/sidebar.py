import streamlit as st

# local imports
from callbacks import slide_seg
from widgets import module_sahi, show_width_lines

def show_sidebar():
    with st.sidebar:
        module_sahi()
        show_width_lines()
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
    st.subheader("James Chen (<niche@vt.edu>)")
    st.write(
        """
        School of Animal Sciences, Virginia Tech

        Blacksburg, VA, USA
        """
    )
    st.subheader("Scotty Yang (<scottyyang@vt.edu>)")
    st.write(
        """
        Department of Entomology, Virginia Tech

        Blacksburg, VA, USA
        """
    )

    st.divider()
    # today = datetime.date.today().strftime("%B %d, %Y")
    st.write("Source code: [Click here](https://github.com/Niche-Lab/ant-detective/tree/main/app)")
    st.write("Last updated:", "June 2025")
