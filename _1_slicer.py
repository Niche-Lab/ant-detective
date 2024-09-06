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


"""



import os
import numpy as np
from dotenv import load_dotenv
import cv2
import tqdm
load_dotenv(".env")
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.getenv('DIR_DATA')
DIR_STUDY = os.path.join(DIR_DATA, "study2")
DIR_DENSE_VAL = os.path.join(DIR_STUDY, "val")
DIR_DENSE_TEST = os.path.join(DIR_STUDY, "test")


# Example usage:
image_folder = "path_to_images"
label_folder = "path_to_labels"
output_image_folder = "path_to_output_images"
output_label_folder = "path_to_output_labels"

for slices in [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5)]:
    slice_x, slice_y = slices
    print(f"Slicing images into {slice_x}x{slice_y} patches...")
    slice_image_and_annotations(
        image_folder, 
        label_folder, 
        output_image_folder, 
        output_label_folder, 
        slice_x=slice_x, slice_y=slice_y)
    
    
def slice_image_and_annotations(
    image_folder,  # path to the folder containing the images
    label_folder,  # path to the folder containing the labels
    output_image_folder, # path to the folder where the sliced images will be saved
    output_label_folder, # path to the folder where the sliced labels will be saved
    slice_x=2, slice_y=4,):

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    # List all images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Determine patch sizes
        patch_width = width // slice_x
        patch_height = height // slice_y

        # Load the corresponding label file
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_folder, label_file)

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
                patch_image_path = os.path.join(output_image_folder, patch_image_file)
                
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
                if patch_labels:
                    patch_label_file = patch_image_file.replace('.jpg', '.txt')
                    patch_label_path = os.path.join(output_label_folder, patch_label_file)
                    with open(patch_label_path, 'w') as f:
                        f.writelines(patch_labels)

    print("Processing complete.")
