import torch
import torch.nn as nn
import torchvision.models as models
from data_loading import dataloaders, transforms
import modeling.trainer as trainer
import torchvision
from typing import Any
import numpy as np
import pandas as pd
import torch
import torchvision

import random
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing
from modeling.model import OpenPoseMC, ResNetMC


argparser = argparse.ArgumentParser()
argp.add_argument(
    "function", help="Choose train or evaluate"
)
argparser.add_argument("--model_name", type=str, default="openposeMC")
argparser.add_argument("--model_path", type=str, default=None)
args = argparser.parse_args()


torch.manual_seed(0)
if __name__ == "__main__":
    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(device)
    video_transformer = transforms.VideoFilePathToTensor(
        max_len=None, fps=10, padding_mode="last"
    )
    H, W = 256, 256
    transforms = torchvision.transforms.Compose(
        [
            transforms.VideoResize([H, W]),
            # transforms.VideoRandomHorizontalFlip(),
        ]
    )

    responses = [
            "pelvis_tilt",
            "ankle_angle_l",
            "ankle_angle_r",
            "hip_adduction_r",
            "hip_adduction_l",
        ]

    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = dataloaders.get_mc_data_loaders(
        video_transformer=video_transformer,
        batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        transforms=transforms,
        preload_videos=False,
        labels=responses,
        num_workers=2,
    )
    # TensorBoard training log
    writer = SummaryWriter(log_dir="expt/")
    if args.model_name == "openposeMC":
         model = OpenPoseMC(num_outputs=len(responses), H=H, W=W, device=device, freeze=True)
    elif args.model_name == "resnetMC":
        model = ResNetMC(num_outputs=len(responses), H=H, W=W, device=device, freeze=True)
    else:
        raise ValueError("Model name not recognized")

    if args.function == "train":
        train_config = trainer.TrainerConfig(
            max_epochs=15,
            learning_rate=4e-4,
            num_workers=4,
            writer=writer,
            ckpt_path="expt/params_mc_testing.pt",
        )

        trainer = trainer.Trainer(
            model=model,
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            config=train_config,
            val_dataloader=val_dl,
            median_freq_weights=False,
            freeze=True
        )
        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        for epoch in range(train_config.max_epochs):
            print(epoch)
            train_losses.append(trainer.train(split="train", step=epoch))
            val_loss = trainer.train(split="val", step=epoch)
            val_losses.append(val_loss)
            print("Val loss:", val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model")
            torch.save(model.state_dict(), f"mc_model_{epoch}.params")
        # write csv of losses
        with open("mc_loss.csv", "w") as f:
            for train_loss, val_loss in zip(train_losses, val_losses):
                f.write(f"{train_loss},{val_loss}\n")

    # test the model
    elif args.function == "evaluate":
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        test_losses = []
        predictions = []
        labels = []
        subj_ids = []

        # create csv of predictions
        for i, (subj_id, x, y) in enumerate(test_dl):
            print(i)
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_losses.append(model.loss_fn(pred, y).item())
            predictions.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            subj_ids.append(subj_id[0])
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        df = pd.DataFrame(predictions, columns=responses)
        df["subj_id"] = subj_ids
        df["label"] = labels
        df.to_csv("mc_predictions.csv")
        print("Test loss:", np.mean(test_losses))
    else:
        raise ValueError("Function not recognized")