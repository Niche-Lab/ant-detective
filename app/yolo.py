from ultralytics import YOLO
import streamlit as st
import os
import shutil
import pandas as pd
import base64

"""
yolo
    /labels
        t1-A1_14_4x10_8_2.txt
        t1-A1_14_4x10_8_3.txt
        ...
    /images
        t1-A1_14_4x10_8_2.jpg
        t1-A1_14_4x10_8_3.jpg
        ...
    /counts
        t1-A1_14_4x10_8_2.jpg
        t1-A1_14_4x10_8_3.jpg
        ...
"""
def predict():
    ls_img = st.session_state.file_imgs
    dir_img = os.path.dirname(ls_img[0])
    model = YOLO("ant_detective.pt")
    model.predict(
        dir_img, 
        save=True,
        show_conf=False,
        line_width=1,
        show_labels=False,
        save_txt=True,
        project="yolo",
        name=".",
    )
    if not os.path.exists("yolo/images"):
        os.makedirs("yolo/images", exist_ok=True)
    if not os.path.exists("yolo/counts"):
        os.makedirs("yolo/counts", exist_ok=True)
    img_count = 0
    for f in os.listdir("yolo"):
        if f.endswith(".jpg"):
            mv_src = os.path.join("yolo", f)
            mv_dst = os.path.join("yolo/counts", f)
            os.rename(mv_src, mv_dst)
            st.session_state.file_pred[img_count] = mv_dst
            img_count += 1
    # copy
    for f in os.listdir("cache"):
        if f.endswith(".jpg"):
            mv_src = os.path.join("cache", f)
            mv_dst = os.path.join("yolo/images", f)
            shutil.copy(mv_src, mv_dst)
    # list counts table
    ls_filenames = []
    ls_counts = []
    for f in os.listdir("yolo/labels"):
        f = f.split(".")[0]
        path_labels = os.path.join("yolo/labels", f + ".txt")
        with open(path_labels, "r") as file:
            lines = file.readlines()
            count = len(lines)
        ls_filenames += [f]
        ls_counts += [count]
    data = pd.DataFrame({"Image": ls_filenames, "Count": ls_counts})
    print(data)
    data.to_csv("yolo/counts.csv", index=False)        

def create_download_zip(
        dir_zip="yolo",
        name_zip="yolo",
):
    """ 
        zip_directory (str): path to directory  you want to zip 
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(name_zip, 'zip', dir_zip)
    # get basename
    filename = os.path.basename(name_zip)
    with open(name_zip + ".zip", 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'ant-detective-{filename}.zip\'>\
            Download file \
        </a>'
        st.markdown(href, unsafe_allow_html=True)