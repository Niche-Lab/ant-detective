import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("out/conf25/results_eval_conf25.csv")
# data = data.loc[:, ["study", "precision", "recall", "split", "n", "model"]]
# data = data.query("split != 'test_b03' and study == 1 and model == 'yolov8n'")
# data_plot = data.melt(id_vars=["study", "split", "n", "model"], var_name="metric", value_name="score")
# # contain a0 in split
# data_plot.loc[:, "similar"] = data_plot["split"].str.contains("a0")
# data_sum = data.groupby(["study", "split", "n", "model"]).agg(["mean", "std"])
# sns.set("notebook")
# g = sns.FacetGrid(
#     data_plot,
#     row="metric",
#     col="similar",
#     # col_wrap=2,
#     margin_titles=True,
#     # sharey="col",
# )
# g.map_dataframe(sns.lineplot, 
#     x="n", y="score", 
#     hue="split", style="split",
#     # hue_order=splits_s1,
#     # style_order=splits_s1,
#     err_style="band", errorbar=("se", 4),
#     markers=True,
#     # palette=["Grey", "#FF1F5B", "#00B000", "#009ADE", "#AF58BA"],
#     )
# # title
# # g.figure.suptitle("Model Generalization in each Data Configuration")
# g.set(
#     xscale="log",
#     xticks=[2**i for i in range(6, 11)],
#     xticklabels=[2**i for i in range(6, 11)],
#     xlabel="Number of training images (n)",)
# g.figure.subplots_adjust(right=1.2)
# g.add_legend()
# g.figure.set_size_inches(8, 6)



data = pd.read_csv("out/conf25/results_sum_conf25.csv")
data = data.query("model == 'yolov8n'").loc[:, ["split", "n", "pre_mean", "pre_std", "rec_mean", "rec_std"]]
# turn to % round to 2 decimal places
data["pre_mean"] = (data["pre_mean"] * 100).round(2)
data["pre_std"] = (data["pre_std"] * 100).round(2)
data["rec_mean"] = (data["rec_mean"] * 100).round(2)
data["rec_std"] = (data["rec_std"] * 100).round(2)
# doulbe std to get 95% confidence interval
data["pre_std"] *= 1.96
data["rec_std"] *= 1.96
# rename columns
data