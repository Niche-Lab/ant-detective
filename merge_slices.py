import os
import cv2
import numpy as np

# Function to merge images back
def merge_images_and_labels(image_dir, 
                            label_dir, 
                            prefix, 
                            slice_size, 
                            output_image_path, 
                            output_label_path,):
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    

    # Gather image slices and labels
    image_slices = {}
    label_slices = {}
    print(prefix)
    for filename in os.listdir(image_dir):
        if filename.startswith(prefix) and filename.endswith('.jpg'):            
            # filename: t1-A1_1_4x10_0_3.jpg
            # Extract row, column coordinates from filename
            _, _, _, row, col = filename.split('.')[0].split('-')[1].split('_')

            # Load image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            image_slices[(row, col)] = img

    for filename in os.listdir(label_dir):
        if filename.startswith(prefix) and filename.endswith('.txt'):
            # Extract row, column coordinates from filename
            _, _, _, row, col = filename.split('.')[0].split('-')[1].split('_')

            # Load label
            label_path = os.path.join(label_dir, filename)
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            label_slices[(row, col)] = label_data

    # Get image size from one of the slices
    slice_height, slice_width = next(iter(image_slices.values())).shape[:2]
    total_height = slice_height * slice_size[0]
    total_width = slice_width * slice_size[1]

    # Create an empty image for merging
    merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    # Merge image slices
    for (row, col), img_slice in image_slices.items():
        r, c = map(int, (row, col))
        y_offset = r * slice_height
        x_offset = c * slice_width
        merged_image[y_offset:y_offset + slice_height, x_offset:x_offset + slice_width] = img_slice

    # Save the merged image
    cv2.imwrite(output_image_path, merged_image)

    # Merge labels and adjust coordinates
    with open(output_label_path, 'w') as f_output:
        for (row, col), labels in label_slices.items():
            r, c = map(int, (row, col))
            for label in labels:
                class_id, x_center, y_center, width, height = map(float, label.split())

                # Adjust the YOLO label coordinates
                x_center = (x_center * slice_width + c * slice_width) / total_width
                y_center = (y_center * slice_height + r * slice_height) / total_height
                width = width * slice_width / total_width
                height = height * slice_height / total_height

                # Write the adjusted label
                f_output.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
