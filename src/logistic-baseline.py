import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

vars_of_interest = {"Age": float, "Height": float, "Weight": float, "Sex": int}
response_var = {"y_fall_risk": int}

def get_rf_dataset(filepath: str, x_vars: dict, y_var: dict) -> pd.DataFrame:
    X = pd.read_csv(filepath, usecols=x_vars.keys(), dtype=x_vars)
    y = pd.read_csv(filepath, usecols=y_var.keys(), dtype=y_var)
    return X, np.array(y).reshape(-1)

X_train, y_train = get_rf_dataset("data/processed/train-survey-data.csv",
                                  x_vars=vars_of_interest,
                                  y_var=response_var)
X_test, y_test = get_rf_dataset("data/processed/test-survey-data.csv",
                                x_vars=vars_of_interest,
                                y_var=response_var)
ids = pd.read_csv("data/processed/test-survey-data.csv", usecols=["subjectid"])

weights = np.array([1, 2, 3])
weights_dict = {label: weight for label, weight in enumerate(weights)}
np.savetxt("data/weights.txt", weights)

logistic = LogisticRegressionCV(class_weight=weights_dict,
                                max_iter=1000, random_state=231)
logistic.fit(X_train, y_train)
preds_test = logistic.predict_proba(X_test)
results = np.hstack((ids, preds_test, y_test.reshape(-1, 1)))
pd.DataFrame(
    results,
    columns=["subjectid", "prob_0", "prob_1", "prob_2", "y"]
).to_csv("predictions/logistic-predictions.csv", index=False)
