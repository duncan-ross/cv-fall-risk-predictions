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

class ResnetTransformer(torch.nn.Module):
    def __init__(self, num_outputs, H, W, hidden_size=1024, num_heads=8, num_layers=6):
        super(ResnetTransformer, self).__init__()
        resnet_net = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.num_outputs = num_outputs

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=num_heads),
            num_layers=num_layers,
            norm=nn.LayerNorm(2048)
        )

        # Linear layers
        self.linear1 = nn.Linear(2048, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.num_outputs)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, median_freq_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Video tensor of shape (N x C x H x W)
            targets (torch.Tensor): Label tensor of shape (N x L)
            median_freq_weights (torch.Tensor): Weights tensor of shape (num_classes,)
        Returns:
            torch.Tensor: Output tensor of shape (N x L x num_outputs)
            torch.Tensor: Loss tensor
        """
        N, C, L, H, W = x.shape
        x = x.view(N * L, C, H, W)

        # Pass the input through the backbone and apply the transformer encoder
        with torch.no_grad():
            x = self.backbone(x)
        x = x.view(N, L, -1)
        x = self.transformer_encoder(x)

        # Now we have a tensor of shape (N, L, -1)
        x = x.view(N * L, -1)
        x = self.linear1(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.relu(x)
        output = self.linear2(x)

        loss = None
        if targets is not None:
            # Reshape output and targets to match the loss function requirements
            output = output.view(N, L, -1)
            targets = targets.view(N * L)

            if median_freq_weights is not None:
                loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output.view(-1, self.num_outputs), targets)
            else:
                loss = torch.nn.CrossEntropyLoss()(output.view(-1, self.num_outputs), targets)

        # Apply softmax to get probabilities for each frame
        output = torch.nn.functional.softmax(output, dim=2)

        return output, loss



torch.manual_seed(0)
if __name__ == '__main__':
    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(device)
    video_transformer = transforms.VideoFilePathToTensor(max_len=None, fps=50, padding_mode='last')
    H, W = 256, 256
    transforms = torchvision.transforms.Compose([
                transforms.VideoResize([H, W]),
                transforms.VideoRandomHorizontalFlip(),
            ])
                                
    # get the dataloaders. can make test and val sizes 0 if you don't want them
    train_dl, val_dl, test_dl = dataloaders.get_mc_data_loaders(
        video_transformer=video_transformer,
        batch_size=4,
        val_batch_size=1,
        test_batch_size=1,
        transforms=transforms,
        preload_videos=False,
        labels=['pelvis_tilt', 'ankle_angle_l','ankle_angle_r','hip_adduction_r','hip_adduction_l'],
        num_workers=2
    )
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=5,
            learning_rate=2e-5, 
            num_workers=4, writer=writer, ckpt_path='expt/params_mc_testing.pt')

    # for subj_id, videos, labels in train_dl:
    #     print(subj_id, videos.size(), labels)
    #     continue

    model = ResnetTransformer(num_outputs=5, H=H, W=W)
    trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, 
                              
    test_dataloader=test_dl, config=train_config, val_dataloader=val_dl, median_freq_weights=False)
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    for epoch in range(train_config.max_epochs):
        print(epoch)
        train_losses.append(trainer.train(split='train', step=epoch))
        val_loss = trainer.train(split='val', step=epoch)
        val_losses.append(val_loss)
        print("Val loss:", val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    # write csv of losses
    with open("mc_loss.txt", 'w') as f:
        for train_loss, val_loss in zip(train_losses, val_losses):
            f.write(f"{train_loss},{val_loss}\n")