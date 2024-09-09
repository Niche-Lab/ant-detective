"""
original whole image
    DIR_DATA, "processed", "b01-dense-fire-ant", "t1-A1_14.JPEG"
sliced images
    DIR_DATA, "paper", "b01-dense", "t1-A1_14_4x10_8_2.jpg"
predicted slices
    DIR_SRC, "runs", "detect", "b01-dense", "images_slices", "t1-A1_14_4x10_8_2.txt"
merged images
    DIR_SRC, "runs", "detect", "b01-dense", "images", "t1-A1_14.jpg"
"""

from ultralytics import YOLO
from dotenv import load_dotenv
import os
import cv2
from merge_slices import merge_images_and_labels

load_dotenv(".env")
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.getenv("DIR_DATA")
DIR_SLICE_SRC = os.path.join(DIR_DATA, "processed", "b01-dense-fire-ant")
DIR_SLICE_DST = os.path.join(DIR_DATA, "paper", "b01-dense")
DIR_SLICE_OUT = os.path.join(DIR_SRC, "runs", "detect", "b01-dense")
            
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


# slice_images(DIR_SLICE_SRC, DIR_SLICE_DST, 4, 10)
model_s2 = YOLO("out/yolo8n_study2.pt")
# took 9m 30s
# model_s2.predict(DIR_SLICE_DST, device="mps", 
#                 save=True,
#                 show_conf=False,
#                 line_width=1,
#                 show_labels=False,
#                 save_txt=True,
#                 name="b01-dense",)
 


ls_imgs = [f for f in os.listdir(DIR_SLICE_SRC) if f.endswith(".JPEG")]
ls_imgs = [f.split('.')[0] for f in ls_imgs]

# Call the function to merge images and labels
for prefix in ls_imgs:
    merge_images_and_labels(
        os.path.join(DIR_SLICE_OUT, "images_slices"),
        os.path.join(DIR_SLICE_OUT, "labels_slices"),
        prefix + "_", (10, 4),
        os.path.join(DIR_SLICE_OUT, "images", prefix + '.jpg'),
        os.path.join(DIR_SLICE_OUT, "labels", prefix + '.txt'),)




def read_counts(dir_labels):
    ls_txt = os.listdir(dir_labels)
    ls_txt = [os.path.join(dir_labels, f) for f in ls_txt]
    ls_txt.sort()
    ls_filenames = [os.path.splitext(f)[0] for f in ls_txt]
    ls_filenames = [os.path.basename(f) for f in ls_filenames]
    ls_counts = []
    for f in ls_txt:
        with open(f, "r") as file:
            lines = file.readlines()
            ls_counts += [len(lines)]
    return ls_counts, ls_filenames


# calculate r^2 and rmse
def r2_rmse(df):
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(df["obs"], df["pre"])
    rmse = mean_squared_error(df["obs"], df["pre"], squared=False)
    rmspe = rmse / df["obs"].mean()
    return r2, rmse, rmspe


dir_pre_s1 = os.path.join(DIR_SRC, "runs", "detect")
dir_obs_s1 = os.path.join(DIR_DATA, "study1")
ls_test_s1 = ["test_a01", "test_a02", "test_a03", "test_b01", "test_b02"]
for t in ls_test_s1:
    dir_t = os.path.join(dir_obs_s1, t, "images")
    model.predict(dir_t, device="mps", 
                save=True,
                show_conf=False,
                line_width=2,
                show_labels=False,
                save_txt=True,
                name=t,)
    
import pandas as pd
# prediction table
# obs, pre, study, split
df_s1 = pd.DataFrame(columns=["obs", "pre", "filename", "study", "split"])
for t in ls_test_s1:
    ls_obs, ls_obs_f = read_counts(os.path.join(dir_obs_s1, t, "labels"))
    df_obs = pd.DataFrame({"obs": ls_obs, "filename": ls_obs_f})
    
    ls_pre, ls_pre_f = read_counts(os.path.join(dir_pre_s1, t, "labels"))
    df_pre = pd.DataFrame({"pre": ls_pre, "filename": ls_pre_f})

    df_temp = pd.merge(df_obs, df_pre, on="filename", how="inner")
    df_temp["study"] = "s1"
    df_temp["split"] = t
    df_s1 = pd.concat([df_s1, df_temp])

import matplotlib.pyplot as plt
import seaborn as sns
# scatter plot
# x = obs, y = pre, hue = split
sns.set("notebook")
sns.scatterplot(data=df_s1, 
                x="obs", y="pre", hue="split",
                s=100, alpha=0.5)
# draw a diagonal line
plt.plot([-10, 100], [-10, 100], color="black", 
         linestyle="--", 
         alpha=0.5, 
         linewidth=1)
plt.xlim(-3, 33)
plt.ylim(-3, 38)
# dpi = 300
plt.savefig("out/plot_s1_n0_30.png", dpi=300)



# sns.scatterplot(data=df_s1.query("obs > 30 and obs <= 60"), 
sns.scatterplot(data=df_s1, 
                x="obs", y="pre", hue="split",
                # dot size
                s=100, alpha=0.5)
# draw a diagonal line
plt.plot([-10, 100], [-10, 100], color="black", 
         linestyle="--", 
         alpha=0.5, 
         linewidth=1)
plt.xlim(28, 60)
plt.ylim(28, 60)
plt.savefig("out/plot_s1_n30_60.png", dpi=300)

# (0.9728376582789352, 0.9061065515567411, 0.1686920568704137)
r2_rmse(df_s1.query("obs <= 30"))
# (0.9471569260238606, 2.798809270624444, 0.06726110396694257)
r2_rmse(df_s1.query("obs > 30"))