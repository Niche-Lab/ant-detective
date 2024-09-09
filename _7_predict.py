from ultralytics import YOLO
from dotenv import load_dotenv
import os
os.getcwd()

load_dotenv(".env")
DIR_SRC = os.getenv('DIR_SRC')
DIR_DATA = os.getenv("DIR_DATA")
model_s1 = YOLO("out/yolo8n_study1.pt")
model_s2 = YOLO("out/yolo8n_study2.pt")

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