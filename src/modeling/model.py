import torch
import torchvision
import torch.nn as nn
from typing import Any

from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd


class BaseVideoModel(torch.nn.Module):
    """
    Base class for video models. Takes in video of (C x L x H x W) and outputs 
    a single vector of length D. Uses conv layers to extract features from
    video and then applies a linear layer to get output vector.
    Conv-> Relu -> Pool -> Concat -> Linear -> Relu -> Linear
    """
    def __init__(self, num_outputs: int, L: int, H: int, W: int):
        super(BaseVideoModel, self).__init__()
        self.num_outputs = num_outputs
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.linear1 = torch.nn.Linear(16 * H//2 * W//2, 16)
        self.linear2 = torch.nn.Linear(16, num_outputs)

    
    def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
        Returns:
            torch.Tensor: Output vector of length D, Loss
        """
        # For each frame, apply conv layer
        N, C, L, H, W = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Apply linear layers to get one vector per video
        x = x.reshape(N, L, -1)
        x = x.mean(dim=1)  # take the mean over the frames to get one vector per video
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        output = x

        loss = None
        if targets is not None:
            # cross entropy loss- only can do with one output column. targets as int of shape (N,)
            targets = targets.reshape(-1).long()
            if median_freq_weights is not None:
                loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
            else:
                loss = torch.nn.CrossEntropyLoss()(output, targets)
        # softmax but do not do gradient
        with torch.no_grad():
            output = torch.nn.functional.softmax(output, dim=1)
        return output, loss


class ResnetLSTM(torch.nn.Module):
    def __init__(self, num_outputs: int, L: int, H: int, W: int):
        super(ResnetLSTM, self).__init__()
        self.num_outputs = num_outputs
        resnet_net = torchvision.models.resnet18(weights="DEFAULT")
        modules = list(resnet_net.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.lstm = torch.nn.LSTM(512, 512, batch_first=True, bidirectional=True)

        # decoder for lstm fc layers
        self.layer1 = nn.Linear(512*2, 256)
        self.layer2 = nn.Linear(256, num_outputs)
        
    
    def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
        Returns:
            torch.Tensor: Output vector of length D, Loss
        """
        N, C, L, H, W = x.shape
        x = x.view(-1, C, H, W)
        # apply resnet backbone but do not change weights
        with torch.no_grad():
            x = self.backbone(x)
        x = x.reshape(N, L, -1)
        x, hidden = self.lstm(x)
        x = x.mean(dim=1)  # take the mean over the frames to get one vector per video
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        output = self.layer2(x)

        loss = None
        if targets is not None:
            targets = targets.reshape(-1).long()
            if median_freq_weights is not None:
                loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
            else:
                loss = torch.nn.CrossEntropyLoss()(output, targets)
        # softmax but do not do gradient
        with torch.no_grad():
            output = torch.nn.functional.softmax(output, dim=1)
        return output, loss


class ResnetTransformer(torch.nn.Module):
        def __init__(self, num_outputs, L, H, W, hidden_size=1024, num_heads=8, num_layers=6):
            super(ResnetTransformer, self).__init__()
            resnet_net = torchvision.models.resnet50(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            self.backbone = torch.nn.Sequential(*modules)


            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=2048, nhead=num_heads),
                num_layers=num_layers,
                norm=nn.LayerNorm(2048)
            )


            # Linear layers
            self.linear1 = nn.Linear(2048*L, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_outputs)

        def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
            Returns:
                torch.Tensor: Output vector of length D, Loss
            """
            N, C, L, H, W = x.shape
            x = x.transpose(1, 2)
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

            # Pass the input through the backbone and apply the transformer encoder
            with torch.no_grad():
                x = self.backbone(x)
            x = x.view(N, L, -1)
            x = self.transformer_encoder(x)
            # Now we have a tensor of shape (N, L, -1)
            x = x.reshape(N, -1)


            x = self.linear1(x)
                        
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            x = torch.relu(x)
            output = self.linear2(x)


            loss = None
            if targets is not None:
                # cross entropy loss- only can do with one output column. targets as int of shape (N,)
                targets = targets.reshape(-1).long()
                if median_freq_weights is not None:
                    loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
                else:
                    loss = torch.nn.CrossEntropyLoss()(output, targets)
            # softmax but do not do gradient
            with torch.no_grad():
                output = torch.nn.functional.softmax(output, dim=1)
            return output, loss


class BaseTransformer(torch.nn.Module):
        def __init__(self, num_outputs, L, H, W, hidden_size=512, num_heads=8, num_layers=6):
            super(ResnetTransformer, self).__init__()
            self.num_outputs = num_outputs
            self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.activation = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))


            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=16 * H//2 * W//2, nhead=num_heads),
                num_layers=num_layers,
                norm=nn.LayerNorm(16 * H//2 * W//2)
            )

            # Global average pooling to reduce the number of features
            self.adaptive_maxpool = nn.AdaptiveMaxPool2d((1, 1))

            # Linear layers
            self.linear1 = nn.Linear(16 * H//2 * W//2*L, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_outputs)

        def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
            Returns:
                torch.Tensor: Output vector of length D, Loss
            """
            N, C, L, H, W = x.shape
            x = x.transpose(1, 2)
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.conv1(x)
            x = self.activation(x)
            x = self.pool(x)
            x = x.view(N, L, -1).permute(1, 0, 2)
            x = self.transformer_encoder(x, )
            x = x.permute(1, 2, 0).contiguous().view(N, -1, 1, 1)

            x = self.adaptive_maxpool(x).view(N, -1)
            x = self.linear1(x)
            x = torch.relu(x)
            output = self.linear2(x)
            # pass to sigmoid to get probabilities
            # output = torch.sigmoid(output)

            # modified_target = torch.zeros_like(output)
            # # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
            # for i in range(modified_target.shape[0]):
            #     modified_target[i,:int(targets[i])+1] = 1
            # loss = (nn.MSELoss(reduction='none')(output, modified_target) * median_freq_weights).sum()

            # output = torch.nn.functional.softmax(output, dim=1)


            loss = None
            if targets is not None:
                # cross entropy loss- only can do with one output column. targets as int of shape (N,)
                targets = targets.reshape(-1).long()
                if median_freq_weights is not None:
                    # binary cross entropy with class weights torch logits
                    loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
                else:
                    loss = torch.nn.CrossEntropyLoss()(output, targets)
            # softmax but do not do gradient
            with torch.no_grad():
                output = torch.nn.functional.softmax(output, dim=1)
            return output, loss
