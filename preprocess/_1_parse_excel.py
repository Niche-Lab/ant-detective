"""
This script is used to extract ant counts from excel spreadsheets.
The excel spreadsheets are from the OHA Foraging Trials Recount Whole Tray.xlsx file.
The output is a csv file with the following columns:
    filename: the image file name
    peptone: the number of ants on the peptone side
    sucrose: the number of ants on the sucrose side
    total: the total number of ants
    trial: the trial number
    split: the split of the data, including train,
"""
import os
import pandas as pd
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
DIR_DATA = os.path.join(ROOT, "data", "raw")
PATH_EXCEL = os.path.join(DIR_DATA, "OHA Foraging Trials Recount Whole Tray.xlsx")
PATH_OUT = os.path.join(DIR_DATA, "peptone_sucrose", "annotations.csv")

# annotations = pd.read_csv(PATH_OUT)
# len(annotations)
# annotations
# traits = ["peptone", "sucrose"]
# labels = annotations.loc[:, traits].values
# labels = labels.astype("float").reshape((-1, len(traits)))
# idx = 3
# os.path.join(os.path.dirname(PATH_OUT), annotations.loc[idx, "filename"])

# annotations.query("split in ['train', 'val']")
# annotations.loc[annotations.split != "test", "split"] = "train"
# annotations.query("split == 'train'").trial.unique()
# np.random.choice([1, 2, 3], 1)[0]


def main():
    os.chdir(ROOT)
    dt_out = make_output_sheet(PATH_EXCEL)
    validate_file(dt_out)
    dt_out.to_csv(PATH_OUT, index=False)


def make_output_sheet(path_excel):
    # initialize the output dataframe
    cols_data = ["filename", "peptone", "sucrose", "total", "trial"]
    dt_out = pd.DataFrame(columns=cols_data)
    # load excel
    # header=3: 4th row is the header
    dict_excel = pd.read_excel(path_excel, sheet_name=None, header=3)
    ls_keys = ["PvS %s" % (i + 1) for i in range(9)]  # PvS 1, PvS 2, ..., PvS 9

    # iterate over sheets
    for i, key in enumerate(ls_keys):
        # first 5 columns, drop 'hours' column
        sheet = dict_excel[key].iloc[:, :5].dropna()
        # format the first column: prefix (G0011, G0022, ...)
        prefix = sheet.columns[0]
        sheet[prefix] = sheet[prefix].astype(int)
        # format the first column values: prefix + number (G0011930, G0022110, ...)
        filename = sheet.apply(lambda row: handle_prefix(prefix, row[prefix]), axis=1)
        # add filename column with its folder name
        sheet["filename"] = [
            os.path.join("Trial %d" % (i + 1), f + ".JPG") for f in filename
        ]
        # subset the selected columns
        sheet = sheet.loc[:, ["filename", "Peptone", "Sucrose", "Total Foragers"]]
        sheet["trial"] = i + 1
        sheet.columns = cols_data
        # append to the output dataframe
        dt_out = pd.concat([dt_out, sheet], axis=0)

    # return the output dataframe
    dt_out = cast_columns(dt_out)
    dt_out = add_split(dt_out)
    return dt_out


def validate_file(dt_out):
    dir_data = os.path.join(ROOT, "data", "raw", "peptone_sucrose")
    for file in dt_out["filename"]:
        try:
            Image.open(os.path.join(dir_data, file))
        except:
            print("File not found:", file)
    print("All files are validated!")


def handle_prefix(prefix, number):
    # exception in trial 6, 8, 9
    # prefix="G0011", number=930  -> "G0011930" (normal)
    # prefix="G0011", number=1010 -> "G0012010"
    # prefix="G0022", number=110  -> "G0022110" (normal)
    # prefix="G0022", number=20   -> "G0022020"
    str_num = str(number)
    if len(str_num) == 2:
        str_num = "0" + str_num
    elif len(str_num) == 4:
        str_num = str_num[1:]
        prefix = prefix[:-1] + str(int(prefix[-1]) + 1)
    return prefix + str_num


def cast_columns(dt_out):
    dt_out["peptone"] = dt_out["peptone"].astype(int)
    dt_out["sucrose"] = dt_out["sucrose"].astype(int)
    dt_out["total"] = dt_out["total"].astype(int)
    dt_out["trial"] = dt_out["trial"].astype(int)
    return dt_out


def add_split(dt_out):
    dt_out["split"] = "train"
    dt_out.loc[dt_out.trial == 6, "split"] = "val"
    dt_out.loc[dt_out.trial > 6, "split"] = "test"
    return dt_out


if __name__ == "__main__":
    main()
