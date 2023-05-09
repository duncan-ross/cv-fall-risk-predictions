import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.nn.functional import cross_entropy

argp = argparse.ArgumentParser()
argp.add_argument("--predictions_file", type=str, required=True)
argp.add_argument("--loss", type=str, default="ce", required=False)
argp.add_argument("--weight_file", type=str, default=None, required=False)

if __name__ == "__main__":
    args = argp.parse_args()
    results = pd.read_csv(args.predictions_file)
    probs = np.array(results.iloc[:, 1:4]) # Middle three columns
    y_true = np.array(results.iloc[:, -1]).reshape(-1)
    y_pred = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if args.loss == "ce":
        weights = np.ones(probs.shape[1], dtype=float)
        acc = accuracy_score(y_true, y_pred)
    elif args.loss == "weighted_ce":
        if args.weight_file is None:
            raise ValueError("loss of weighted_ce, but no weight file provided")
        weights = np.loadtxt(args.weight_file, dtype=float)
        num = (np.diag(conf_mat) * weights).sum()
        den = (np.bincount(y_true) * weights).sum()
        acc = num / den
    else:
        e = "loss argument must be one of 'ce' (cross entropy) "
        e += "or 'weighted_ce (weighted cross entropy)"
        raise ValueError(e)
    print(weights)
    ce = cross_entropy(
        input=torch.FloatTensor(probs),
        target=torch.LongTensor(y_true),
        weight=torch.FloatTensor(weights)
    )
    print(f"(Weighted) Cross-Entropy Loss on Test Set: {ce}")
    print(f"(Weighted) Accuracy on Test Set: {acc}")
    print(f"Confusion Matrix of Results:\n{conf_mat}")

