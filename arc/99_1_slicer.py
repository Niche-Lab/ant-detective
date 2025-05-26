"""
This script slice the dataset into smaller patches

prerequisite: YOLO annotation from Roboflow. The train path must be modified by removing the relative path ".."

study2
    data.yaml
    train
        images
        labels
    test <input>
        images
        labels
    val <output> patch images for test
    test_patch <output> patch images for test


step 1:
    slice image in the test folder
step 2:
    move one set of images to the val folder
"""

import os
from dotenv import load_dotenv
import cv2
import tqdm
load_dotenv(".env")
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.getenv('DIR_DATA')
DIR_STUDY = os.path.join(DIR_DATA, "study2")
DIR_DENSE_VAL = os.path.join(DIR_STUDY, "val")
DIR_DENSE_TEST = os.path.join(DIR_STUDY, "test")
LS_SLICES = [(2, 2), (2, 4), (4, 4), (4, 10), (8, 10)]

def main():
    for slices in LS_SLICES:
        slice_x, slice_y = slices
        print(f"Slicing images into {slice_x}x{slice_y} patches...")
        slice_images(DIR_DENSE_TEST, DIR_DENSE_TEST + "_%sx%s" % (slice_x, slice_y),
                    slice_x=slice_x, slice_y=slice_y)
    mv_to_val(LS_SLICES)
    modify_yaml(LS_SLICES)

def modify_yaml(ls_slices):
    with open(os.path.join(DIR_STUDY, "data.yaml"), "w") as f:
        f.write("path: %s\n" % DIR_STUDY)
        f.write("nc: 1\n")
        f.write("names: ['ant']\n")
        f.write(f"train: {DIR_STUDY}/train\n")
        f.write(f"val: {DIR_STUDY}/val\n")
        f.write(f"test: {DIR_STUDY}/test\n")
        for slices in ls_slices:
            slice_x, slice_y = slices
            f.write(f"test_{slice_x}x{slice_y}: {DIR_STUDY}/test_{slice_x}x{slice_y}\n")

def mv_to_val(ls_slices):
    FILE_A1 = "t1-A1_1_JPEG"
    FILE_A2 = "t2-A2_20_JPEG"
    # destination
    dir_dst = DIR_DENSE_VAL
    dir_img_dst, dir_lb_dst = make_dir(dir_dst)

    # move patches (2x2, 2x4, 4x4, 4x5)
    for slices in ls_slices:
        slice_x, slice_y = slices
        # source
        dir_src = DIR_DENSE_TEST + "_%sx%s" % (slice_x, slice_y)
        dir_img_src = os.path.join(dir_src, "images")
        dir_lb_src = os.path.join(dir_src, "labels")
        # mv images
        for f in os.listdir(dir_img_src):
            if FILE_A1 in f or FILE_A2 in f:
                mv_src = os.path.join(dir_img_src, f)
                mv_dst = os.path.join(dir_img_dst, f)
                os.rename(mv_src, mv_dst)
        # mv labels
        for f in os.listdir(dir_lb_src):
            if FILE_A1 in f or FILE_A2 in f:
                mv_src = os.path.join(dir_lb_src, f)
                mv_dst = os.path.join(dir_lb_dst, f)
                os.rename(mv_src, mv_dst)
    # move test (1x1) images
    dir_img_src = os.path.join(DIR_DENSE_TEST, "images")
    dir_lb_src = os.path.join(DIR_DENSE_TEST, "labels")
    for f in os.listdir(dir_img_src):
        if FILE_A1 in f or FILE_A2 in f:
            mv_src = os.path.join(dir_img_src, f)
            mv_dst = os.path.join(dir_img_dst, f)
            os.rename(mv_src, mv_dst)
    for f in os.listdir(dir_lb_src):
        if FILE_A1 in f or FILE_A2 in f:
            mv_src = os.path.join(dir_lb_src, f)
            mv_dst = os.path.join(dir_lb_dst, f)
            os.rename(mv_src, mv_dst)
    


def make_dir(dir_name):
    dir_img = os.path.join(dir_name, "images")
    dir_lb = os.path.join(dir_name, "labels")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_lb, exist_ok=True)
    return dir_img, dir_lb

def slice_images(dir_src, dir_dst, slice_x, slice_y):
    dir_img_src, dir_lb_src = make_dir(dir_src)
    dir_img_dst, dir_lb_dst = make_dir(dir_dst)

    # List all images
    image_files = [f for f in os.listdir(dir_img_src) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(dir_img_src, image_file)
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Determine patch sizes
        patch_width = width // slice_x
        patch_height = height // slice_y

        # Load the corresponding label file
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(dir_lb_src, label_file)

        with open(label_path, 'r') as f:
            labels = f.readlines()

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
                filename = os.path.splitext(image_file)[0] # remove extension
                patch_image_file = f"{filename}_{slice_x}x{slice_y}_{i}_{j}.jpg"
                patch_image_path = os.path.join(dir_img_dst, patch_image_file)
                
                # Save the sliced image
                cv2.imwrite(patch_image_path, patch)

                # Adjust the annotations for the patch
                patch_labels = []
                for label in labels:
                    class_id, cx, cy, w, h = map(float, label.split())
                    
                    # Denormalize the coordinates
                    abs_cx = cx * width
                    abs_cy = cy * height
                    abs_w = w * width
                    abs_h = h * height

                    # Check if the bounding box is in the current patch
                    if (x_start <= abs_cx <= x_end) and (y_start <= abs_cy <= y_end):
                        # Adjust the bounding box relative to the new patch
                        new_cx = (abs_cx - x_start) / patch_width
                        new_cy = (abs_cy - y_start) / patch_height
                        new_w = abs_w / patch_width
                        new_h = abs_h / patch_height
                        
                        patch_labels.append(f"{int(class_id)} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
                
                # Write new label file if there are any labels in this patch
                patch_label_file = patch_image_file.replace('.jpg', '.txt')
                patch_label_path = os.path.join(dir_lb_dst, patch_label_file)
                if patch_labels:
                    with open(patch_label_path, 'w') as f:
                        f.writelines(patch_labels)
                else:
                    # Still create an empty label file
                    with open(patch_label_path, 'w') as f:
                        pass

if __name__ == "__main__":
    main()