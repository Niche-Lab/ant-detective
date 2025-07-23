import scipy
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = pd.read_csv("local/out/s1_results.csv")
data = data.query("metric == 'map50' & strategy in ['baseline', 'grid_search']")
data = data.loc[:, ["score", "n_samples", "model", "strategy", "similar"]]
data.columns = ["map50", "n_samples", "model", "SAHI", "similar"]
# order the categorical variables
data["model"] = pd.Categorical(data["model"], categories=["yolo11n", "yolo11m", "rtdetr-l"])
data["SAHI"] = pd.Categorical(data["SAHI"], categories=["baseline", "grid_search"])
data["similar"] = pd.Categorical(data["similar"], categories=[True, False])

model = ols("map50 ~ C(model) + C(SAHI) + C(n_samples) + C(similar)", data=data).fit()
model.summary()


anova_results = anova_lm(model, typ=2)
anova_results