import numpy as np
import pandas as pd

# Properly clean target steps, target active time variables
def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.NaN


if __name__ == "__main__":
    survey_datasets = {}
    survey_datasets["train"] = pd.read_csv("data/processed/train-survey-data.csv")
    survey_datasets["val"] = pd.read_csv("data/processed/val-survey-data.csv")
    survey_datasets["test"] = pd.read_csv("data/processed/test-survey-data.csv")
    for grp, survey_data in survey_datasets.items():
        for col in ["targetSteps", "targetActiveTime"]:
            survey_data[col] = survey_data[col]\
                .apply(lambda x: str(x).replace(",", ""))\
                .apply(convert_to_float)
        kept_cols = [
            col for col in survey_data
            if col.find("TEXT") < 0 and
            col not in [
                "subjectid", "Country", "State", "feedback_open", "continue", "difficult_open", "falling_1", "falling_2", "falling_3"
            ]
        ]
        survey_data = survey_data[kept_cols]
        encoding_cols = [
            feat
            for feat in list(survey_datasets["train"].select_dtypes("object"))
            if feat in kept_cols
            and feat not in ("targetSteps", "targetActiveTime")
        ]
        for col in encoding_cols:
            split_col = survey_data[col].apply(lambda x: str(x).split(","))
            unique_vals = survey_datasets["train"][col]\
                .apply(lambda x: str(x).split(",")).explode().unique()
            for val in sorted(unique_vals):
                survey_data[f"{col}_{val}"] = split_col\
                    .apply(lambda x: any([y == val for y in x]))\
                    .astype(int)
            survey_data.drop(columns=col, inplace=True)
        survey_data.to_csv(f"data/processed/{grp}-survey-data.csv", index=False)
    