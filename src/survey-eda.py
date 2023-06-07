# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
survey_data = pd.read_csv("../data/processed/train-survey-data.csv")

# %%
# Three-level fall risk
for col in survey_data.columns:
    if col in ["y_fall_risk", "y_fall_risk_binary"]:
        continue
    heatmap_data = survey_data.value_counts(subset=[col, "y_fall_risk"])\
        .reset_index()\
        .pivot_table(index=col, columns="y_fall_risk", values=0)\
        .sort_index()
    sns.heatmap(heatmap_data, cmap="Blues", annot=True)
    plt.show()


