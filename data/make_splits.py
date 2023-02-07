"""
This script is used to split the data into train, val, and test sets. The outputs include two parts: images and annotation.txt
"""
import os
import pandas as pd
import numpy as np
from PIL import Image

WD = "/Users/niche/Library/CloudStorage/OneDrive-VirginiaTech/_03_Papers/find_ants/"
DIR_ROOT = os.path.join(WD, "data", "peptone_sucrose")
RATIO = 5  # reduce size by 1/RATIO

os.chdir(DIR_ROOT)

# image --------------------
for i in range(9):
    os.chdir(os.path.join(DIR_ROOT, "raw", f"Trial {i + 1}"))
    ls_imgs = os.listdir()
    for image in ls_imgs:
        img = Image.open(image)
        npimg = np.array(img)[::RATIO, ::RATIO, :]
        name_out = "t%d_" % (i + 1) + image[5:]
        if i < 5:
            Image.fromarray(npimg).save(os.path.join(DIR_ROOT, "train", name_out))
        elif i == 5:
            Image.fromarray(npimg).save(os.path.join(DIR_ROOT, "val", name_out))
        else:
            Image.fromarray(npimg).save(os.path.join(DIR_ROOT, "test", name_out))


# labels --------------------
def process_single_filename(filename):
    # 30.JPG -> 030.JPG
    # 132.JPG -> 132.JPG
    # 1053.JPG -> 053.JPG
    filename = str(int(filename)) + ".JPG"
    if len(filename) == 6:
        filename = "0" + filename
    elif len(filename) == 7:
        filename = filename
    elif len(filename) == 8:
        filename = filename[1:]
    return filename


def process_filename(filenames, trial):
    filenames = [process_single_filename(f) for f in filenames]
    filenames = ["t%s_" % (trial) + f for f in filenames]
    return filenames


path_meta = os.path.join(WD, "data", "OHA Foraging Trials.xlsx")
dict_meta = pd.read_excel(path_meta, sheet_name=None)

col_out = ["filename", "time", "sucrose", "peptone", "total_foragers", "trial", "split"]
col_keep = ["Time", "Sucrose", "Peptone", "Total Foragers"]
df_out = pd.DataFrame(columns=col_out)
ls_sheets = ["P v S %d" % (i + 1) for i in range(9)]
for sheet in ls_sheets:
    t = str(sheet[-1])
    data_sub = pd.concat(
        [dict_meta[sheet].iloc[:, 0], dict_meta[sheet].loc[:, col_keep]], axis=1
    ).dropna()
    # handling filenames
    data_sub[data_sub.columns[0]] = process_filename(
        filenames=data_sub[data_sub.columns[0]], trial=t
    )
    # trial information
    data_sub["trial"] = "trial_" + t
    if int(t) < 6:
        data_sub["split"] = "train"
    elif int(t) == 6:
        data_sub["split"] = "val"
    else:
        data_sub["split"] = "test"
    # columns
    data_sub.columns = col_out
    # concat each sheet
    df_out = pd.concat([df_out, data_sub], axis=0)

# convert types
df_out["time"] = df_out["time"].astype(str)
df_out["sucrose"] = df_out["sucrose"].astype(int)
df_out["peptone"] = df_out["peptone"].astype(int)
df_out["total_foragers"] = df_out["total_foragers"].astype(int)
df_out["trial"] = df_out["trial"].astype(str)
df_out["split"] = df_out["split"].astype(str)

for s in ["train", "val", "test"]:
    df_out.query("split == @s").to_csv(
        os.path.join(DIR_ROOT, s, "annotation.txt"), index=False
    )
