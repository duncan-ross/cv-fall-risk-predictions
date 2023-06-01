import argparse
from data_loading import dataloaders, transforms
import torchvision
from modeling.trainer import calculate_weights
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.nn.functional import cross_entropy

argp = argparse.ArgumentParser()
argp.add_argument("--predictions_file", type=str, required=True)
argp.add_argument("--loss", type=str, default="weighted_ce", required=False)

if __name__ == "__main__":
    args = argp.parse_args()
    results = pd.read_csv(args.predictions_file)
    print(f"Evaluating on {args.predictions_file}")
    probs = np.array(results.iloc[:, 1:4])  # Middle three columns
    y_true = np.array(results.iloc[:, -1]).reshape(-1)
    y_pred = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if args.loss == "ce":
        weights = np.ones(probs.shape[1], dtype=float)
        acc = accuracy_score(y_true, y_pred)
    elif args.loss == "weighted_ce":
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        video_transformer = transforms.VideoFilePathToTensor(
            max_len=35, fps=5, padding_mode="last"
        )
        transforms = torchvision.transforms.Compose(
            [
                transforms.VideoResize([256, 256]),
                transforms.VideoRandomHorizontalFlip(),
            ]
        )
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
        weights = calculate_weights(train_dl, device).cpu().numpy()
        num = (np.diag(conf_mat) * weights).sum()
        den = (np.bincount(y_true) * weights).sum()
        acc = num / den
    else:
        e = "loss argument must be one of 'ce' (cross entropy) "
        e += "or 'weighted_ce (weighted cross entropy)"
        raise ValueError(e)
    print(f"Weights: {weights}")
    N = probs.shape[0]
    l = -weights[y_true] * np.log(probs[np.arange(N), y_true])
    ce = l.sum() / weights[y_true].sum()
    print(f"(Weighted) Cross-Entropy Loss on Test Set: {ce}")
    print(f"(Weighted) Accuracy on Test Set: {acc}")
    print(f"Confusion Matrix of Results:\n{conf_mat}")
    print("Proportion for Each Risk Level:")
    print(conf_mat / conf_mat.sum(axis=1).reshape(-1, 1))
