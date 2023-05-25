from data_loading import dataloaders, transforms
from modeling.trainer import calculate_weights
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import torch
import torchvision


vars_of_interest = {"Age": float, "Height": float, "Weight": float, "Sex": int}
response_var = {"y_fall_risk": int}


def get_rf_dataset(filepath: str, x_vars: dict, y_var: dict) -> pd.DataFrame:
    X = pd.read_csv(filepath, usecols=x_vars.keys(), dtype=x_vars)
    y = pd.read_csv(filepath, usecols=y_var.keys(), dtype=y_var)
    return X, np.array(y).reshape(-1)


X_train, y_train = get_rf_dataset(
    "data/processed/train-survey-data.csv", x_vars=vars_of_interest, y_var=response_var
)
X_test, y_test = get_rf_dataset(
    "data/processed/test-survey-data.csv", x_vars=vars_of_interest, y_var=response_var
)
ids = pd.read_csv("data/processed/test-survey-data.csv", usecols=["subjectid"])

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
video_transformer = transforms.VideoFilePathToTensor(
    max_len=35, fps=5, padding_mode="last"
)

H, W = 256, 256
transforms = torchvision.transforms.Compose([
                transforms.VideoResize([H, W]),
                # transforms.VideoRandomHorizontalFlip(),
            ])

train_dl, _, _ = dataloaders.get_vid_data_loaders(
    video_transformer=video_transformer,
    batch_size=4,
    val_batch_size=1,
    test_batch_size=1,
    transforms=transforms,
    preload_videos=False,
    labels=["y_fall_risk"],
    num_workers=0,
)
weights = calculate_weights(train_dl, device).numpy()
weights_dict = {label: weight for label, weight in enumerate(weights)}

logistic = LogisticRegressionCV(
    class_weight=weights_dict, max_iter=1000, random_state=231
)
logistic.fit(X_train, y_train)
preds_test = logistic.predict_proba(X_test)
results = np.hstack((ids, preds_test, y_test.reshape(-1, 1)))
pd.DataFrame(results, columns=["subjectid", "prob_0", "prob_1", "prob_2", "y"]).to_csv(
    "predictions/logistic-predictions.csv", index=False
)
