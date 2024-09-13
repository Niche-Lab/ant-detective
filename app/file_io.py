import os
import pandas as pd
import shutil
import streamlit as st
import datetime
import base64

def clean_up(dir_src="cache", dir_dst="yolo"):
    shutil.rmtree(dir_src, ignore_errors=True)
    shutil.rmtree(dir_dst, ignore_errors=True)
    for f in os.listdir():
        if f.endswith(".zip"):
            os.remove(f)
            
    
def inspect_results(dir_src="cache", dir_dst="yolo"):
    """
    <dir_src>
        img1.jpg
        img2.jpg
        ...
    <dir_dst>
        images/ - original images
        labels/ - yolo format labels
        counts/ - vidsualized detections
        counts.csv - counts table <filename, count>    
    """
    dir_images = os.path.join(dir_dst, "images")
    dir_labels = os.path.join(dir_dst, "labels")
    dir_counts = os.path.join(dir_dst, "counts")
    path_csv = os.path.join(dir_dst, "counts.csv")
    
    # check folder
    for d in [dir_images, dir_labels, dir_counts]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # mv detected images to counts/ folder
    img_count = 0
    for f in os.listdir(dir_dst):
        if f.endswith(".jpg"):
            mv_src = os.path.join(dir_dst, f)
            mv_dst = os.path.join(dir_counts, f)
            os.rename(mv_src, mv_dst)
            st.session_state.file_pred[img_count] = mv_dst
            img_count += 1
    # copy original images to images/ folder
    for f in os.listdir(dir_src):
        if f.endswith(".jpg"):
            mv_src = os.path.join(dir_src, f)
            mv_dst = os.path.join(dir_images, f)
            shutil.copy(mv_src, mv_dst)
    # list counts table
    ls_filenames = []
    ls_counts = []
    for f in os.listdir(dir_labels):
        f = f.split(".")[0]
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

