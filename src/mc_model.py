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
from modeling.model import OpenPoseMC


class ResnetModel(torch.nn.Module):
    def __init__(self, num_outputs, H, W, hidden_size=256, num_heads=2, num_layers=2):
        super(ResnetModel, self).__init__()
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.num_outputs = num_outputs

        # Linear layers
        self.linear1 = nn.Linear(512, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.num_outputs)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None,
        median_freq_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Frames tensor of shape (C x N x H x W)
            targets (torch.Tensor): Label tensor of shape (N x num_outputs)
            median_freq_weights (torch.Tensor): Weights tensor of shape (num_classes,)
        Returns:
            torch.Tensor: Output tensor of shape (N x num_outputs)
            torch.Tensor: Loss tensor
        """
        C, N, H, W = x.shape
        # N X C X H X W -> N X H X W X C
        x = x.transpose(0, 1)

        # Pass the input through the backbone and apply the transformer encoder
        with torch.no_grad():
            x = self.backbone(x)

        # Now we have a tensor of shape (N, -1)
        x = x.view(N, -1)
        x = self.linear1(x)
        # x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.relu(x)
        output = self.linear2(x)

        loss = None
        if targets is not None:
            # MSE
            loss = torch.nn.functional.mse_loss(output, targets)
        print(output, loss)
        return output, loss


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
        batch_size=4,
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

    # model = ResnetModel(num_outputs=5, H=H, W=W)
    model = OpenPoseMC(num_outputs=5, H=H, W=W, device=device)
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
    # write csv of losses
    with open("mc_loss.csv", "w") as f:
        for train_loss, val_loss in zip(train_losses, val_losses):
            f.write(f"{train_loss},{val_loss}\n")
