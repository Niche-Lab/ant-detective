import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv('.env')
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.path.join(os.getenv('DIR_DATA'), "final")
os.chdir(DIR_DATA)

def main():
    ls_splits = ["train", "val", 
                "test_b01", "test_b03", "test_b04", "test_b05", "test_b06", 
                "test_a01", "test_a02", "test_a03"]
    columns=["split", "prefix", "yyyymmdd", "HHMM", "datetime", "count", 
            "filename", "path_img", "path_txt"]

    df = pd.DataFrame(columns=columns)
    for split in ls_splits:
        filenames = [f[:-4] for f in os.listdir(os.path.join(split, "labels"))]
        filenames.sort()
        for filename in filenames:
            # filename example: 
            # test_b01: t1-A1_7_JPEG.rf.c37d2e0aba1538efc44d759dda81a5c2.txt
            # others: t1-20221109-1347_jpg.rf.f881f682ca1ac5809ef25.txt
            
            # get datetime
            if split == "test_b01":
                prefix, yyyymmdd, HHMM = filename.split("_")
                prefix = prefix.split("-")[1]
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
                            pd.DataFrame([[split, prefix, yyyymmdd, HHMM, datetime, count, filename, path_img, path_txt]], 
                                        columns=columns)])
            
    df.to_csv(os.path.join(DIR_SRC, "metadata.csv"), index=False)
    # make summary
    df_agg = df.\
        groupby("split").\
            agg({"count": ["mean", "std", "count", "median", "min", "max", "sum"]}).\
            reset_index()
    df_agg.columns = ["split", "mean", "std", "count", "median", "min", "max", "sum"]
    df_agg.to_csv(os.path.join(DIR_SRC, "summary.csv"), index=False)
    
if __name__ == "__main__":
    main()
