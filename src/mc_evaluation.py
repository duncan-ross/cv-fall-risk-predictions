import argparse
from data_loading import dataloaders, transforms
import torchvision
from modeling.trainer import calculate_weights
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.nn.functional import cross_entropy
from settings import MC_RESPONSES

argp = argparse.ArgumentParser()
argp.add_argument("--predictions_file", type=str, default="predictions/openpose_mc_predictions.csv")

if __name__ == "__main__":
    args = argp.parse_args()
    results = pd.read_csv(args.predictions_file)
    pred_cols = [f'pred_{i}' for i in MC_RESPONSES]

    # Get the predictions
    y_pred = results[pred_cols].values
    y_true = results[MC_RESPONSES].values

    # get MSE for each response and compare to baseline of guessing the mean
    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    baseline_mse = np.mean((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    print("MSE:", mse)
    print("Baseline MSE:", baseline_mse)
    print("MSE improvement:", (baseline_mse - mse)/baseline_mse)

    # Get the max and min prediction for each subject
    max_preds = results.groupby("subj_id").apply(lambda x: x[pred_cols].max(axis=0))
    min_preds = results.groupby("subj_id").apply(lambda x: x[pred_cols].min(axis=0))
    # see if the ordering of them is the same as the ordering of the true labels
    true_max = results.groupby("subj_id").apply(lambda x: x[MC_RESPONSES].max(axis=0))
    true_min = results.groupby("subj_id").apply(lambda x: x[MC_RESPONSES].min(axis=0))
    print(true_max)
    print(true_min)
    print(max_preds)
    print(min_preds)
    