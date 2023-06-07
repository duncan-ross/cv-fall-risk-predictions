# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportions_ztest

# %%
losses = pd.read_csv(
    "../data/losses.txt", header=None, names=["Training", "Validation"]
)
losses["Epoch"] = np.arange(start=1, stop=len(losses) + 1)
losses = losses.melt(id_vars="Epoch", var_name="Data", value_name="Loss")
losses = losses[losses.Epoch <= 6]

# %%
sns.set_theme(palette=["darkred", "darkgreen"], style="white")
sns.lineplot(losses, x="Epoch", y="Loss", hue="Data")
plt.xticks(ticks=np.arange(1, 11))
plt.vlines(x=4,
           ymin=losses.Loss.min(),
           ymax=losses[losses.Data == "Validation"].Loss.min(),
           linestyles="dashed",
           colors="grey")
plt.title("Training and Validation Loss for Fusion Model")
plt.savefig("../visuals/loss-plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
test = pd.read_csv("../data/processed/test-survey-data.csv")
test_predictions = pd.read_csv("../predictions/fusion_mc_preds.csv")

# %%
test_predictions["subjectid"] = test_predictions.id\
    .apply(lambda x: re.findall("(?<=').*(?=')", x)[0])
probs = np.array(test_predictions.iloc[:, 1:4])  # Middle three columns
y_true = np.array(test_predictions.iloc[:, -2]).reshape(-1)
y_pred = np.argmax(probs, axis=1)
test_predictions["correct"] = y_true == y_pred

# %%
merged_results = pd.merge(
    left=test, right=test_predictions[["subjectid", "correct"]],
    how="left", on="subjectid"
)

# %%
feature_plots = {
    "Age": "boxplot",
    "Height": "boxplot",
    "Sex": "heatmap",
    "Education": "heatmap",
    "Income": "heatmap",
    "OA_check": "heatmap",
    "Fall_YN": "heatmap",
    "PAtracker": "heatmap",
    "difficult_scale": "heatmap",
    "amm_3": "heatmap",
    "amm_4": "heatmap",
    "amm_5": "heatmap",
    "SEE_2": "heatmap",
    "SEE_3": "heatmap",
    "SC9": "heatmap",
    "Global01": "barplot",
    "Global03": "barplot"
}

# %%
for feature, plot_type in feature_plots.items():
    if plot_type == "heatmap":
        heatmap_data = merged_results.value_counts(subset=["correct", feature])\
            .reset_index()\
            .pivot_table(index=feature, columns="correct", values=0)\
            .sort_index()
        sns.heatmap(heatmap_data, cmap="Blues", annot=True)
    elif plot_type == "boxplot":
        sns.boxplot(merged_results, x="correct", y=feature)
    elif plot_type == "barplot":
        sns.countplot(merged_results, x=feature, hue="correct")
    elif plot_type == "histogram":
        sns.histplot(merged_results, x=feature, hue="correct")
    plt.show()

# %%
sns.set_theme(palette=sns.color_palette("deep")[2:4][::-1], style="white")
merged_results["Correctly Classified?"] = [
    "Yes" if res else "No" for res in merged_results.correct
]

# %%
# Age Plot
sns.boxplot(merged_results, x="Correctly Classified?", y="Age")
plt.xlabel("Correctly Classified?")
plt.ylabel("Age")
plt.title("Age Distributions vs. Whether Correctly Classified")
plt.savefig("../visuals/age-plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
sex_split = merged_results.value_counts(["Sex", "Correctly Classified?"])\
    .reset_index().rename(columns={0: "N"})
sex_split.Sex = sex_split.Sex.map({1: "Female", 0: "Male"})
sex_split.loc[sex_split.Sex == "Female", "N"] = (
    sex_split.loc[sex_split.Sex == "Female", "N"] /
    sex_split.loc[sex_split.Sex == "Female", "N"].sum()
)
sex_split.loc[sex_split.Sex == "Male", "N"] = (
    sex_split.loc[sex_split.Sex == "Male", "N"] /
    sex_split.loc[sex_split.Sex == "Male", "N"].sum()
)

# %%
# Sex Plot
sns.barplot(sex_split, x="Sex", y="N",
            hue="Correctly Classified?", hue_order=["No", "Yes"])
plt.ylim(0, 1)
plt.xlabel("Sex")
plt.ylabel("Proportion")
plt.title("Sex Distribution by Whether Correctly Classified")
plt.text(x=0.2, y=0.6, s="18 / 31", ha="center", va="center")
plt.text(x=1.2, y=0.675, s="17 / 26", ha="center", va="center")
plt.savefig("../visuals/sex-plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Follow up test for significance
successes = merged_results[merged_results["Correctly Classified?"] == "Yes"]\
    .value_counts("Sex").sort_index().to_numpy()
n_obs = merged_results.value_counts("Sex").sort_index().to_numpy()
z_stat, p_val = proportions_ztest(successes, n_obs, 0)
print("{0:0.3f}".format(p_val))

# %%
# Health Plot
sns.countplot(merged_results, x="Global03", hue="Correctly Classified?")
plt.xlabel("Physical Health Rating")
plt.ylabel("Frequency")
plt.title("Physical Health Rating Distribution by Whether Correctly Classified")
plt.savefig("../visuals/health-plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Confusion Matrix
conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
sns.heatmap(np.round(conf_mat / conf_mat.sum(axis=1).reshape(-1, 1), 2),
            vmin=0, vmax=1,
            cmap="Greens", linewidths=0.5, annot=True)
plt.xticks([0.5, 1.5, 2.5], labels=["Low", "Medium", "High"])
plt.yticks([0.5, 1.5, 2.5], labels=["Low", "Medium", "High"])
plt.xlabel("Predicted Fall Risk")
plt.ylabel("Actual Fall Risk")
plt.title("Confusion Matrix for Fall Risk\nNormalized by Actual Risk")
plt.savefig("../visuals/conf-mat.png", dpi=300, bbox_inches="tight")
plt.show()


