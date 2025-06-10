import os
import pandas as pd
import shutil
import streamlit as st
import datetime
import base64

def clean_up(dir_src="cache"):
    shutil.rmtree(dir_src, ignore_errors=True)
    for f in os.listdir():
        if f.endswith(".zip"):
            os.remove(f)            

def inspect_results(dir_dst="output"):
    """
    Move the images and generated labels to the respective folders
    compress the folder and return the image count
    
    <dir_dst>
        images/ - original images
        labels/ - yolo format labels
        counts/ - vidsualized detections
        counts.csv - counts table <filename, count>    
    """
    dir_counts = os.path.join(dir_dst, "counts") 
    dir_labels = os.path.join(dir_dst, "labels") # yolo created
    path_csv = os.path.join(dir_dst, "counts.csv")
    
    # mv detected images from <root> to <root>/counts/ folder
    img_count = 0
    for f in os.listdir(dir_counts):
        if is_img(f):     
            mv_dst = os.path.join(dir_counts, f)
            st.session_state.file_pred[img_count] = mv_dst
            img_count += 1

    # list counts table
    ls_filenames = []
    ls_counts = []
    for f in os.listdir(dir_labels):
        f = os.path.splitext(f)[0]
        path_labels = os.path.join(dir_labels, f + ".txt")
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
    shutil.make_archive("ant-detective", 'zip', dir_dst) # ant-detective.zip
    # return
    return img_count   


def is_img(f):
    return f.upper().endswith(".JPG") or\
           f.upper().endswith(".JPEG") or\
           f.upper().endswith(".PNG")
