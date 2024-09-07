import os
from dotenv import load_dotenv
import pandas as pd
from pyniche.data.yolo.API import YOLO_API

load_dotenv('.env')
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA_STUDY1 = os.getenv('DIR_DATA_STUDY1')
DIR_DATA_STUDY2 = os.getenv('DIR_DATA_STUDY2')

def main():
    columns=["study", "split", "prefix", "yyyymmdd", "HHMM", "datetime", "count", 
            "filename", "path_img", "path_txt"]
    df = pd.DataFrame(columns=columns)
    api_s1 = YOLO_API(DIR_DATA_STUDY1)
    api_s2 = YOLO_API(DIR_DATA_STUDY2)
    data = dict({"study1": api_s1, "study2": api_s2})
    ls_imgs = data["study1"].splits["test"]["images"]
    ls_imgs = [os.path.splitext(f)[0] for f in ls_imgs]
    
    for study in ["study1", "study2"]:
        if study == "study1":
            os.chdir(DIR_DATA_STUDY1)
        else:
            os.chdir(DIR_DATA_STUDY2)
        for split in data[study].splits:
            # rm extension and basename
            ls_imgs = data[study].splits[split]["images"]
            filenames = [os.path.splitext(f)[0] for f in ls_imgs]
            filenames = [os.path.basename(f) for f in filenames]
            filenames.sort()
            
            for filename in filenames:
                # filename example: 
                # test_b01: t1-A1_7_JPEG.rf.c37d2e0aba1538efc44d759dda81a5c2.txt
                # others: t1-20221109-1347_jpg.rf.f881f682ca1ac5809ef25.txt
                
                # get datetime
                if (study == "study2" and split != "train") or\
                    (study == "study1" and split == "test") or\
                    (study == "study1" and split == "test_b03"):
                    # a special case for dense fire ant
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
    # make summary
    df_agg = df.\
        groupby(["study", "split"]).\
            agg({"count": ["mean", "std", "count", "median", "min", "max", "sum"]}).\
            reset_index()
    df_agg.columns = ["study", "split", "mean", "std", "count", "median", "min", "max", "sum"]
    df_agg.to_csv(os.path.join(DIR_SRC, "summary.csv"), index=False)
    
if __name__ == "__main__":
    main()
