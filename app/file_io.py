import os
import pandas as pd
import shutil
import streamlit as st
from pathlib import Path
import datetime
import base64

def clean_up(dir_src="cache"):
    shutil.rmtree(dir_src, ignore_errors=True)
    for f in os.listdir():
        if f.endswith(".zip"):
            os.remove(f)            

def inspect_results():
    """
    Move the images and generated labels to the respective folders
    compress the folder and return the image count
    
    <dir_dst>
        images/ - original images
        labels/ - yolo format labels
        counts/ - vidsualized detections
        counts.csv - counts table <filename, count>    
    """
    
    FILEPATH = __file__
    DIR_ROOT = Path(os.path.dirname(FILEPATH))
    dir_out = DIR_ROOT / "output"
    dir_counts = dir_out / "counts"
    dir_labels = dir_out / "labels" 
    path_csv = dir_out / "counts.csv"

    # mv detected images from <root> to <root>/counts/ folder
    img_count = 0
    for f in dir_counts.iterdir():
        if is_img(f):
            mv_dst = dir_counts / f
            st.session_state.file_pred[img_count] = mv_dst
            img_count += 1

    # list counts table
    ls_filenames = []
    ls_counts = []
    for f in dir_labels.iterdir():
        f = f.stem
        path_labels = dir_labels / (f + ".txt")
        with open(path_labels, "r") as file:
            lines = file.readlines()
            count = len(lines)
        ls_filenames += [f]
        ls_counts += [count]
    # save counts table
    data = pd.DataFrame({"Image": ls_filenames, 
                         "Count": ls_counts})
    data.to_csv(path_csv, index=False)
    # zip it
    # today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    shutil.make_archive("ant-detective", 'zip', dir_out) # ant-detective.zip
    # return
    return img_count   


def is_img(f):
    # to regular string
    f = str(f)
    return f.upper().endswith(".JPG") or\
           f.upper().endswith(".JPEG") or\
           f.upper().endswith(".PNG")
