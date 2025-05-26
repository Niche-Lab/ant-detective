import os
import sys
import pandas as pd

# local imports
from paths import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API

DIR_SRC = PATHS["DIR_SRC"]
DIR_DATA_STUDY1 = PATHS["DIR_DATA"] / "study1"
DIR_DATA_STUDY2 = PATHS["DIR_DATA"] / "study2"

def main():
    columns=["study", "split", "prefix", "yyyymmdd", "HHMM", "datetime", "count", 
            "filename", "path_img", "path_txt"]
    df = pd.DataFrame(columns=columns)
    api_s1 = YOLO_API(DIR_DATA_STUDY1)
    api_s2 = YOLO_API(DIR_DATA_STUDY2)
    data = dict({"study1": api_s1, "study2": api_s2})
    
    for study in ["study1", "study2"]:
        if study == "study1":
            os.chdir(DIR_DATA_STUDY1)
        else:
            os.chdir(DIR_DATA_STUDY2)
        for split in data[study].splits:
            # rm extension and and only keep the basename
            filenames = [f.stem for f in data[study][split].images]
            filenames.sort()
            
            for filename in filenames:
                # filename example: 
                # test_b01: t1-A1_7_JPEG.rf.c37d2e0aba1538efc44d759dda81a5c2
                # others: t1-20221109-1347_jpg.rf.f881f682ca1ac5809ef25
                
                # get datetime
                if (study == "study2" and split != "train") or\
                    (study == "study1" and split == "test"):
                    # put pseudo prefix for test split (dense ant images)
                    prefix = filename[:2]
                    yyyymmdd = '19991231'
                    HHMM = '2359'
                else:
                    prefix, yyyymmdd, HHMM = filename.split("-")
                    HHMM = HHMM.split("_")[0]
                datetime = pd.to_datetime(yyyymmdd + HHMM, format="%Y%m%d%H%M")
                # paths
                path_img = os.path.join(split, "images", filename + ".jpg")
                path_txt = os.path.join(split, "labels", filename + ".txt")
                # counts
                with open(path_txt, "r") as f:
                    lines = f.readlines()
                    count = len(lines)
                # append to df
                df = pd.concat([df, 
                                pd.DataFrame([[study, split, prefix, yyyymmdd, HHMM, datetime, count, filename, path_img, path_txt]], 
                                        columns=columns)])
            
    df.to_csv(os.path.join(DIR_SRC, "metadata.csv"), index=False)
    print("Successfully saved metadata to %s" % os.path.join(DIR_SRC, "metadata.csv"))
    print("Number of rows in metadata: %d" % len(df))
    # make summary
    df_agg = df.\
        groupby(["study", "split"]).\
            agg({"count": ["mean", "std", "count", "median", "min", "max", "sum"]}).\
            reset_index()
    df_agg.columns = ["study", "split", "mean", "std", "count", "median", "min", "max", "sum"]
    df_agg.to_csv(os.path.join(DIR_SRC, "summary.csv"), index=False)
    print("Successfully saved summary to %s" % os.path.join(DIR_SRC, "summary.csv"))
    print("Number of rows in summary: %d" % len(df_agg))
    print("Done.")
    
if __name__ == "__main__":
    main()
