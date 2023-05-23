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

    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = dataloaders.get_mc_data_loaders(
        video_transformer=video_transformer,
        batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        transforms=transforms,
        preload_videos=False,
        labels=[
            "pelvis_tilt",
            "ankle_angle_l",
            "ankle_angle_r",
            "hip_adduction_r",
            "hip_adduction_l",
        ],
        num_workers=2,
    )
    # TensorBoard training log
    writer = SummaryWriter(log_dir="expt/")

    train_config = trainer.TrainerConfig(
        max_epochs=15,
        learning_rate=2e-4,
        num_workers=4,
        writer=writer,
        ckpt_path="expt/params_mc_testing.pt",
    )

    model = ResNetMC(num_outputs=5, H=H, W=W)
    #model = OpenPoseMC(num_outputs=5, H=H, W=W, device=device)
    trainer = trainer.Trainer(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        config=train_config,
        val_dataloader=val_dl,
        median_freq_weights=False,
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
