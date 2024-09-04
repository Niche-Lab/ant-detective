from multiprocessing import process
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime
import os

load_dotenv('../.env')

DIR_SRC = os.getenv('DIR_SRC')
DIR_RAW = os.path.join(DIR_SRC, 'data', 'raw')
DIR_DST = os.path.join(DIR_SRC, 'data', 'processed')
HEIGHT = 1080
if not os.path.exists(DIR_DST):
    os.makedirs(DIR_DST)

os.chdir(DIR_SRC)
os.listdir(DIR_RAW)

# get mtime and to yyyymmdd-hhmm
def get_mtime(ls_files):
    mtime = [os.path.getmtime(f) for f in ls_files]
    mtime = [datetime.fromtimestamp(m).strftime('%Y%m%d-%H%M') for m in mtime]
    return mtime

def create_dir(dir_create):
    if not os.path.exists(dir_create):
        os.makedirs(dir_create)

def process_trial_based(names, n_trials, prefix, resize=True, rename=True):
    create_dir(os.path.join(DIR_DST, names["dst"]))
    for i in range(n_trials):
        prefix_src = prefix + str(i + 1)
        prefix_dst = "t%d-" % (i + 1)
        process(names, prefix_src, prefix_dst, resize, rename)
       

def process_virus_based(names, ls_prefix, resize=True, rename=True):
    create_dir(os.path.join(DIR_DST, names["dst"]))
    for prefix_src, prefix_dst in zip(ls_prefix, ["ctrl-", "virus-"]):
        process(names, prefix_src, prefix_dst, resize, rename) 

def process(names, prefix_src, prefix_dst, 
            resize=True, rename=True):
    dir_src = os.path.join(DIR_RAW, names["src"], prefix_src)
    files_src = [os.path.join(dir_src, f) for f in os.listdir(dir_src) if ".JPG" in f or ".JPEG" in f]
    if rename:
        # whether use meta change time as new name
        files_dst = get_mtime(files_src)
        files_dst = [f + ".jpg" for f in files_dst]
    else:
        # keep the original basename
        files_dst = [os.path.basename(f) for f in files_src]
    for f_src, f_dst in zip(files_src, files_dst):
        img = Image.open(f_src)
        if resize:
            # resize to 1080px height
            img = img.resize((int(HEIGHT * img.width / img.height), HEIGHT))
        img.save(os.path.join(DIR_DST, names["dst"], prefix_dst + f_dst))
            



# CONTROL VS VIRUS
# a01: Bait Trials
names = dict({
    "src": "Bait Trials",
    "dst": "a01-bait-trials",
})
process_virus_based(names, ls_prefix=["Control", "1762 virus"])

# a02: Virus Honey Trials
names = dict({
    "src": "Virus Honey Trials",
    "dst": "a02-virus-honey-trials",
})
process_virus_based(names, ls_prefix=["1_OHA_007_Control", "1_OHA_007_1762"])

# a03: Virus Sugar Trials
names = dict({
    "src": "Virus Peptone _ Sucrose Trials",
    "dst": "a03-virus-peptone-sucrose-trials",
})
process_virus_based(names, ls_prefix=["1_OHA_007_Control", "1_OHA_007_1762"])

# MULTI-TRIALS
# b01: fire_ant_activity
names = dict({
    "src": "fire_ant_activity",
    "dst": "b01-dense-fire-ant"
})
process_trial_based(names, n_trials=2, prefix="A", 
                    resize=False, rename=False)

# b02: honey_trials
names = dict({
    "src": "honey_trials",
    "dst": "b02-honey-trials"
})
process_trial_based(names, n_trials=4, prefix="OHA honey-lemon trial ")

# b03: honey_trials_v2
names = dict({
    "src": "honey_trials_v2",
    "dst": "b03-honey-trials-v2"
})
process_trial_based(names, n_trials=11, prefix="Trial ")

# b04: peptone_sucrose
names = dict({
    "src": "peptone_sucrose",
    "dst": "b04-peptone-sucrose"
})
process_trial_based(names, n_trials=9, prefix="Trial ")

# b05: sugar_trials
names = dict({
    "src": "sugar_trials",
    "dst": "b05-sugar-trials"
})
process_trial_based(names, n_trials=4, prefix="OHA sugar trial ")

# b06: test_trials
names = dict({
    "src": "test_trials",
    "dst": "b06-test-trials"
})
process_trial_based(names, n_trials=3, prefix="OHA sugar test trial ")