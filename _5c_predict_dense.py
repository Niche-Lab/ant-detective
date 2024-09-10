"""
Input images
    DIR_DATA, 
        "processed", "b01-dense-fire-ant", "t1-A1_14.JPEG" # ORIGINAL IMAGE
        "paper", "b01-dense", "t1-A1_14_4x10_8_2.jpg" # SLICED IMAGE

Predicted slices
    DIR_SRC/runs/detect/b01-dense
                    images_slices/"t1-A1_14_4x10_8_2.jpg"
                    images/"t1-A1_14.jpg"
"""

from ultralytics import YOLO
from dotenv import load_dotenv
import os
import cv2
from merge_slices import merge_images_and_labels

load_dotenv(".env")
DIR_DATA = os.getenv("DIR_DATA")
DIR_SRC = os.getenv('DIR_SRC')
DIR_ORIGINAL = os.path.join(DIR_DATA, "processed", "b01-dense-fire-ant")
DIR_SLICE_IN = os.path.join(DIR_DATA, "paper", "b01-dense")
DIR_SLICE_OUT = os.path.join(DIR_SRC, "runs", "detect", "b01-dense")
DIR_SLICE_IMAGES = os.path.join(DIR_SLICE_OUT, "images_slices")
DIR_SLICE_LABELS = os.path.join(DIR_SLICE_OUT, "labels_slices")
DIR_MERGED_IMAGES = os.path.join(DIR_SLICE_OUT, "images")
DIR_MERGED_LABELS = os.path.join(DIR_SLICE_OUT, "labels")

def main():
    slice_images(DIR_ORIGINAL, DIR_SLICE_IN, 4, 10)
    predict_slices()
    rearange_slices()
    merge_slices()

def predict_slices():
    """
    output
    ---

    DIR_SRC/runs/detect/b01-dense
        labels/
            t1-A1_14_4x10_8_2.txt
            t1-A1_14_4x10_8_3.txt
            ...
        t1-A1_14_4x10_8_2.jpg
        t1-A1_14_4x10_8_3.jpg
        ...
    """
    model_s2 = YOLO(os.path.join(DIR_SRC, "out", "yolo8n_study2.pt"))
    # took 9m 30s
    model_s2.predict(DIR_SLICE_IN, device="mps", 
                    save=True,
                    show_conf=False,
                    line_width=1,
                    show_labels=False,
                    save_txt=True,
                    name="b01-dense",)

def rearange_slices():
    """
    1. rename labels/ to labels_slices/
    2. move *.jpg to images_slices/*.jpg

    DIR_SRC/runs/detect/b01-dense
        labels_slices/
            t1-A1_14_4x10_8_2.txt
            t1-A1_14_4x10_8_3.txt
            ...
        images_slices/
            t1-A1_14_4x10_8_2.jpg
            t1-A1_14_4x10_8_3.jpg
            ...
    """
    os.makedirs(DIR_MERGED_IMAGES, exist_ok=True)
    os.makedirs(DIR_SLICE_LABELS, exist_ok=True)
    os.makedirs(DIR_SLICE_IMAGES, exist_ok=True)
    # mv labels/*.txt to labels_slices/*.txt
    for f in os.listdir(dir_labels):
        if f.endswith(".txt"):
            mv_src = os.path.join(DIR_MERGED_LABELS, f)
            mv_dst = os.path.join(DIR_SLICE_LABELS, f)
            os.rename(mv_src, mv_dst)
    # mv *.jpg to images_slices/*.jpg
    for f in os.listdir(DIR_SLICE_OUT):
        if f.endswith(".jpg"):
            mv_src = os.path.join(DIR_SLICE_OUT, f)
            mv_dst = os.path.join(DIR_SLICE_IMAGES, f)
            os.rename(mv_src, mv_dst)



def merge_slices():
    ls_imgs = [f for f in os.listdir(DIR_ORIGINAL) if f.endswith(".JPEG")]
    ls_imgs = [f.split('.')[0] for f in ls_imgs]
    for prefix in ls_imgs:
        merge_images_and_labels(
            DIR_SLICE_IMAGES,
            DIR_SLICE_LABELS,
            prefix + "_", (10, 4),
            os.path.join(DIR_MERGED_IMAGES, prefix + '.jpg'),
            os.path.join(DIR_MERGED_LABELS, prefix + '.txt'),)


def slice_images(dir_src, dir_dst, slice_x, slice_y):
    os.makedirs(dir_dst, exist_ok=True)
    ls_img = [f for f in os.listdir(dir_src) if f.endswith(".JPEG")]
    for img in ls_img:
        img_path = os.path.join(dir_src, img)
        slice_image(img_path, dir_dst, slice_x, slice_y)
    
def slice_image(img_path, dir_dst, slice_x, slice_y):
    # Load the image
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Determine patch sizes
    patch_width = width // slice_x
    patch_height = height // slice_y
    # Process the image in slices
    for i in range(slice_y):
        for j in range(slice_x):
            # Extract patch coordinates
            x_start = j * patch_width
            y_start = i * patch_height
            x_end = x_start + patch_width
            y_end = y_start + patch_height

            # Slice the image
            patch = img[y_start:y_end, x_start:x_end]
            
            # Create new filenames
            filename = os.path.splitext(img_path)[0] # remove extension
            filename = os.path.basename(filename)
            patch_image_file = f"{filename}_{slice_x}x{slice_y}_{i}_{j}.jpg"
            patch_image_path = os.path.join(dir_dst, patch_image_file)
            # Save the sliced image
            cv2.imwrite(patch_image_path, patch)
