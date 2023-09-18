"""
This script match the original excel filename to the annotated filenames. And generate the split files for HuggingFace.

- Original images: matching filename with the label.
- Grid images: extract label from the filename.

Prerequisite:
- There must be processed (grided) images in grid and original folders.

"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DIR_DATA = os.path.join(ROOT, "data")
DIR_OUT = os.path.join(DIR_DATA, "ant-pep-suc")
DIR_RAW = os.path.join(DIR_DATA, "raw", "peptone_sucrose")
PATH_RAW_ANNT = os.path.join(DIR_RAW, "annotations.csv")
DIR_IMG_ORI = os.path.join(DIR_OUT, "original")
DIR_IMG_GRID = os.path.join(DIR_OUT, "grid")

# matching raw filename with the label
df_out = pd.DataFrame(columns=["name_raw", "id"])
for t in range(1, 10):
    dir_trial = os.path.join(DIR_RAW, "Trial %d" % t)
    names_raw = [f for f in os.listdir(dir_trial) if ".JPG" in f]
    df_tmp = pd.DataFrame(
        {
            "name_raw": names_raw,
            "id": ["t%d_%d" % (t, i + 1) for i in range(len(names_raw))],
        }
    )
    # modify filename Gxxx.JPG -> Trial t/Gxxx.JPG
    df_tmp["name_raw"] = df_tmp.apply(
        lambda row: "Trial %d/%s" % (t, row["name_raw"]), axis=1
    )
    # append to df_out
    df_out = pd.concat([df_out, df_tmp], axis=0)

# merge df_out and df_raw
df_raw = pd.read_csv(PATH_RAW_ANNT)
df_raw.columns = [
    "name_raw",
    "manual_peptone",
    "manual_sucrose",
    "manual_all",
    "trial",
    "split",
]
df_out = pd.merge(df_out, df_raw, on="name_raw", how="left")

# Grid version ---------------------------------------------------------------
# extract info from the filename: t1_100_tl_02.jpg
# <id>_<grid>_<count>.jpg
df_grid_tmp = pd.DataFrame(columns=["filename", "id", "grid", "n"])
ls_grid = [f for f in os.listdir(DIR_IMG_GRID) if ".jpg" in f]
for f in ls_grid:
    # segments
    s1, s2, s3, s4 = f.split("_")  # t1, 100, tl, 02.jpg
    # append to df_grid_tmp
    tmp = pd.DataFrame(
        {
            "filename": f,
            "id": s1 + "_" + s2,
            "grid": s3,
            "n": int(s4.split(".")[0]),
        },
        index=[0],
    )
    df_grid_tmp = pd.concat([df_grid_tmp, tmp], axis=0)
df_grid = pd.merge(df_out.copy(), df_grid_tmp, on="id", how="left")

# Original version -----------------------------------------------------------
# group by id to count n
df_ori_tmp = df_grid.groupby("id").agg({"n": "sum"}).reset_index()
df_ori_tmp["filename"] = df_ori_tmp.apply(lambda row: row["id"] + "_0.jpg", axis=1)
df_ori = pd.merge(df_out.copy(), df_ori_tmp, on="id", how="left")

# organize and save
df_grid = df_grid.loc[
    :,
    [
        "n",
        "trial",
        "filename",
        "id",
        "manual_peptone",
        "manual_sucrose",
        "manual_all",
        "split",
    ],
]
df_ori = df_ori.loc[
    :,
    [
        "n",
        "trial",
        "filename",
        "id",
        "manual_peptone",
        "manual_sucrose",
        "manual_all",
        "split",
    ],
]
df_grid.to_csv(os.path.join(DIR_OUT, "grid.csv"), index=False)
df_ori.to_csv(os.path.join(DIR_OUT, "original.csv"), index=False)

# export splits
for split in ["train", "val", "test"]:
    df_grid.query("split == '%s'" % split).to_csv(
        os.path.join(DIR_OUT, "grid_%s.csv" % split), index=False
    )
    df_ori.query("split == '%s'" % split).to_csv(
        os.path.join(DIR_OUT, "original_%s.csv" % split), index=False
    )
