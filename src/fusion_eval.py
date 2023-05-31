import numpy as np
import pandas as pd
import torch
import torchvision

import random
import argparse

from modeling import trainer, model
from data_loading import dataloaders, transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing
from modeling.trainer import calculate_weights
import os
from sklearn.metrics import accuracy_score, confusion_matrix



torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument(
    "--fusion_tuning_path",
    type=str,
    help="Which dataset to use (train/val/test)",
    default="tuning/fusion/",
    required=False,
)
argp.add_argument(
    "--mc_model_path",
    type=str,
    help="Path to the reading params file",
    default="model/best_model2.params",
)
argp.add_argument(
    "--mc_model_type",
    type=str,
    help="openposeMC/resnetMC",
    default="openposeMC",
)
argp.add_argument(
    "--fps",
    type=int,
    help="FPS used in training",
    default=2,
    required=False,
)
args = argp.parse_args()

def prepare_model(mc_model_path: str, dataset: str, fps: int):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(device)
    video_transformer = transforms.VideoFilePathToTensor(
        max_len=fps*30, fps=fps, padding_mode="zero"
    )
    H, W = 256, 256
    extra_transforms = torchvision.transforms.Compose(
        [
            transforms.VideoResize([H, W]),
        ]
    )
    train_dl, val_dl, test_dl = dataloaders.get_fusion_data_loaders(
            video_transformer=video_transformer,
            batch_size=1,
            val_batch_size=1,
            test_batch_size=1,
            transforms=extra_transforms,
            preload_videos=False,
            num_workers=2,
        )
    class_weights = calculate_weights(train_dl, device).cpu().numpy()
    dl = {"train": train_dl, "val": val_dl, "test": test_dl}
    dl = dl[dataset]
    fusion_model = model.FusionModel(num_features=123, num_outputs=3, num_mc_outputs=5, 
    mc_model_type="openposeMC", mc_model_path=mc_model_path, device=device)
    return fusion_model, dl, device, class_weights

# Foward through the fusion model- keep the losses
def evaluate_model(model, dl, class_weights, device, fusion_model_path):
    model.load_state_dict(torch.load(fusion_model_path))
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    losses = []
    pred_cols = [
            'pred_fall_risk_0', 'pred_fall_risk_1', 'pred_fall_risk_2'
        ]
    actual_cols = ['y_fall_risk']
    predictions = []
    pbar = tqdm(enumerate(dl), total=len(dl))
    class_weights = torch.tensor(class_weights, device=device)
    for it, (subj_id, x, y) in pbar:
        y = y.to(device)
        print(it)
        # place data on the correct device
        with torch.no_grad():
            pred, loss = model(x, y, class_weights)
            print(pred)
            print(y)
            predictions.append(
                (
                    {
                        "id": subj_id,
                        **dict(zip(pred_cols, pred.tolist()[0])),
                        **dict(zip(actual_cols, y.tolist()[0])),
                    }
                )
            )
            losses.append(loss.item())

    preds_df = pd.DataFrame(predictions)
    return np.mean(losses), preds_df

def get_weighted_accuracy(preds_df, class_weights):
    probs = np.array(preds_df.iloc[:, 1:4])  # Middle three columns
    y_true = np.array(preds_df.iloc[:, -1]).reshape(-1)
    y_pred = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    num = (np.diag(conf_mat) * class_weights).sum()
    den = (np.bincount(y_true) * class_weights).sum()
    acc = num / den
    return acc

def main():
    # Loop through fusion_tuning_path and find all the best_model.params
    # For each best_model.params, load the model and evaluate it
    # Save the results in a csv
    model, dl, device, class_weights = prepare_model(mc_model_path=args.mc_model_path, dataset="val", fps=args.fps)
    # Loop through the fusion_tuning_path and find all the best_model.params
    # It could be nested in a folder. Generally we have a folder for each trial
    # within the fusion_tuning_path folder and each one has a best_model.params
    best_acc = 0
    best_loss = 1e9
    best_model_path = ""
    for root, dirs, files in os.walk(args.fusion_tuning_path):
        for file in files:
            if file.endswith(".params"):
                fusion_model_path = os.path.join(root, file)
                # Evaluate the model
                loss, preds_df = evaluate_model(model, dl, class_weights, device, fusion_model_path)
                acc = get_weighted_accuracy(preds_df, class_weights)
                if acc > best_acc:
                    # for now, we only care about accuracy
                    best_acc = acc
                    best_loss = loss
                    best_model_path = fusion_model_path
                    preds_df.to_csv("best_val_preds.csv", index=False)
    # Evaluate the best model
    print("Best model path on val set: ", best_model_path)
    print("Best accuracy on val set: ", best_acc)
    print("Loss on val set: ", best_loss)

    # Evaluate the best model on test set
    model, dl, device, class_weights = prepare_model(mc_model_path=args.mc_model_path, dataset="test", fps=args.fps)
    loss, preds_df = evaluate_model(model, dl, class_weights, device, best_model_path)
    acc = get_weighted_accuracy(preds_df, class_weights)
    print("Accuracy on test set: ", acc)
    print("Loss on test set: ", loss)
    preds_df.to_csv("best_test_preds.csv", index=False)

if __name__ == "__main__":
    main()
