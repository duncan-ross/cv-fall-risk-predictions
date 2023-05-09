import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.nn.functional import cross_entropy

argp = argparse.ArgumentParser()
argp.add_argument("--predictions_file", type=str, required=True)

if __name__ == "__main__":
    args = argp.parse_args()
    results = pd.read_csv(args.predictions_file)
    probs = np.array(results.iloc[:, 2:5]) # Middle three columns
    y_true = np.array(results.iloc[:, -1]).reshape(-1)
    y_pred = np.argmax(probs, axis=1)
    ce = cross_entropy(
        input=torch.Tensor(y_pred),
        target=torch.Tensor(y_true).type(torch.LongTensor)
    )
    print(f"Cross-Entropy Loss on Test Set: {ce}")
    print(f"Accuracy on Test Set: {accuracy_score(y_true, y_pred)}")
    print(f"Confusion Matrix of Results:")
    print(confusion_matrix(y_true=y_true, y_pred=y_pred))
