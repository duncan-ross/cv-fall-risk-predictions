import torch
import torchvision
import torch.nn as nn
from typing import Any

from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

import numpy as np
from PIL import Image
from pytorch_openpose.src.model import bodypose_model
from pytorch_openpose.src import util
from data_loading import transforms
from settings import BODY_MODEL_PATH


class BaseVideoModel(torch.nn.Module):
    """
    Base class for video models. Takes in video of (C x L x H x W) and outputs 
    a single vector of length D. Uses conv layers to extract features from
    video and then applies a linear layer to get output vector.
    Conv-> Relu -> Pool -> Concat -> Linear -> Relu -> Linear
    """
    def __init__(self, num_outputs: int, L: int, H: int, W: int, device='cpu'):
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
            final_output = torch.nn.functional.softmax(output, dim=1)
        return final_output, loss


class ResnetLSTM(torch.nn.Module):
    def __init__(self, num_outputs: int, L: int, H: int, W: int, device='cpu'):
        super(ResnetLSTM, self).__init__()
        self.num_outputs = num_outputs
        resnet_net = torchvision.models.resnet18(weights="DEFAULT")
        modules = list(resnet_net.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.lstm = torch.nn.LSTM(512, 512, batch_first=True, bidirectional=True)

        # decoder for lstm fc layers
        self.layer1 = nn.Linear(512*2, 512)
        self.layer_norm = nn.LayerNorm(512)
        # self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, num_outputs)
        
    
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
        # apply resnet backbone but do not change weights
        # with torch.no_grad():
        #     x = self.backbone(x)
        x = self.backbone(x)
        x = x.reshape(N, L, -1)
        x, hidden = self.lstm(x)
        x = hidden[0].transpose(0, 1).reshape(N, -1)
        # x = x.reshape(N, -1)
        x = self.layer1(x)
        # x = self.layer_norm(x)
        x = torch.nn.functional.relu(x)
        # x = self.layer2(x)
        # x = torch.nn.functional.relu(x)
        output = self.layer3(x)

        loss = None
        if targets is not None:
            if targets.shape[1] == 1:
                # cross entropy loss- only can do with one output column. targets as int of shape (N,)
                targets = targets.reshape(-1).long()
                if median_freq_weights is not None:
                    # binary cross entropy with class weights torch logits
                    loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
                else:
                    loss = torch.nn.CrossEntropyLoss()(output, targets)
            else:
                # first target binary, rest is regression
                # class weights median freq[0] when class 0, median freq[1] when class 1
                class_weights = median_freq_weights[targets[:, 0].long()]
                # weighting these two losses equally
                loss = 5*torch.nn.BCEWithLogitsLoss(weight=class_weights)(output[:, 0], targets[:, 0])
                loss += torch.nn.MSELoss()(output[:, 1:], targets[:, 1:])
        # softmax but do not do gradient
        if self.num_outputs == 3:
            with torch.no_grad():
                final_output = torch.nn.functional.softmax(output, dim=1)
        else:
            # sigmoid but do not do gradient
            with torch.no_grad():
                final_output = torch.clone(output)
                final_output[:, 0] = torch.sigmoid(final_output[:, 0])
        print(final_output, loss)
        return final_output, loss


class ResnetTransformer(torch.nn.Module):
        def __init__(self, num_outputs, L, H, W, hidden_size=512, num_heads=2, num_layers=2, device='cpu'):
            super(ResnetTransformer, self).__init__()
            resnet_net = torchvision.models.resnet18(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            self.backbone = torch.nn.Sequential(*modules)


            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=num_heads),
                num_layers=num_layers,
                norm=nn.LayerNorm(512)
            )


            # Linear layers
            self.linear1 = nn.Linear(512*L, hidden_size)
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
            x = x.view(N, L, -1).transpose(0, 1)
            x = self.transformer_encoder(x)
            # Now we have a tensor of shape (N, L, -1)
            x = x.transpose(0, 1)
            x = x.reshape(N, -1)


            x = self.linear1(x)
            x = torch.relu(x)
            output = self.linear2(x)


            loss = None
            if targets is not None:
                if targets.shape[1] == 1:
                    # cross entropy loss- only can do with one output column. targets as int of shape (N,)
                    targets = targets.reshape(-1).long()
                    if median_freq_weights is not None:
                        # binary cross entropy with class weights torch logits
                        loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
                    else:
                        loss = torch.nn.CrossEntropyLoss()(output, targets)
                else:
                    # first target binary, rest is regression
                    # class weights median freq[0] when class 0, median freq[1] when class 1
                    class_weights = median_freq_weights[targets[:, 0].long()]
                    # weighting these two losses equally
                    loss = torch.nn.BCEWithLogitsLoss(weight=class_weights)(output[:, 0], targets[:, 0])
                    loss += torch.nn.MSELoss()(output[:, 1:], targets[:, 1:])
            # softmax but do not do gradient
            if targets.shape[1] == 1:
                with torch.no_grad():
                    final_output = torch.nn.functional.softmax(output, dim=1)
            else:
                # sigmoid but do not do gradient
                with torch.no_grad():
                    final_output = torch.clone(output)
                    final_output[:, 0] = torch.sigmoid(final_output[:, 0])
            print(final_output, loss)
            return final_output, loss


class BaseOpenPose(torch.nn.Module):
    def __init__(self, num_outputs, L, H, W, hidden_size=512, num_heads=8, num_layers=6, device='cpu'):
        super(BaseOpenPose, self).__init__()
        self.device = device
        self.model = bodypose_model()
        model_dict = util.transfer(self.model, BODY_MODEL_PATH)
        self.model.load_state_dict(model_dict)
        self.model = self.model
        # modules = list(self.model.children())[:-1]
        # self.backbone = torch.nn.Sequential(*modules)

        # reduce the number of channels
        self.rnn = nn.GRU(input_size=38*23*23, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # Linear layers
        self.fc1 = nn.Linear(4 * 64, num_outputs)

        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
        Returns:
            torch.Tensor: Output vector of length N, Loss
        """
        N, C, L, H, W = x.shape
        x = x.transpose(1, 2)
        out = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        data = transforms.process_image(out)

        out, _ = self.model(data)
        # out is (N*L, 38, 17, 17)
        # COMPLETE CODE TO GET OUTPUT which should be (N) dimensional
        # linear, relu, linear
        out = out.reshape(N, L, -1)
        output, hid = self.rnn(out)

        hid = hid.reshape(N, -1)
        output = self.fc1(hid)
        loss = None

        # compute loss
        if targets is not None:
            if targets.shape[1] == 1:
                # cross entropy loss- only can do with one output column. targets as int of shape (N,)
                targets = targets.reshape(-1).long()
                if median_freq_weights is not None:
                    # binary cross entropy with class weights torch logits
                    loss = torch.nn.CrossEntropyLoss(weight=median_freq_weights)(output, targets)
                else:
                    loss = torch.nn.CrossEntropyLoss()(output, targets)
            else:
                # first target binary, rest is regression
                # class weights median freq[0] when class 0, median freq[1] when class 1
                class_weights = median_freq_weights[targets[:, 0].long()]
                # weighting these two losses equally
                loss = torch.nn.BCEWithLogitsLoss(weight=class_weights)(output[:, 0], targets[:, 0])
                loss += torch.nn.MSELoss()(output[:, 1:], targets[:, 1:])
                
        # output
        if targets.shape[1] == 1:
            with torch.no_grad():
                final_output = torch.nn.functional.softmax(output, dim=1)
        else:
            # sigmoid but do not do gradient
            with torch.no_grad():
                final_output = torch.clone(output)
                final_output[:, 0] = torch.sigmoid(final_output[:, 0])
            print(final_output)
            return final_output, loss


class OpenPoseMC(torch.nn.Module):
    def __init__(self, num_outputs, H, W, hidden_size=512, device='cpu', freeze: bool = True):
        super(OpenPoseMC, self).__init__()
        self.device = device
        self.model = bodypose_model()
        
        model_dict = util.transfer(self.model, torch.load(BODY_MODEL_PATH))
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(device)
        # modules = list(self.model.children())[:-1]
        # self.backbone = torch.nn.Sequential(*modules)


        # Linear layers
        self.fc1 = nn.Linear(38 * 23 * 23, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_outputs)

        self.num_outputs = num_outputs
        self.freeze = freeze

    def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Video tensor of shape (N X C x L x H x W)
        Returns:
            torch.Tensor: Output vector of length N, Loss
        """
        C, N, H, W = x.shape
        # N X C X H X W -> N X H X W X C
        x = x.transpose(0, 1)
        data = transforms.process_image(x).to(self.device)

        if self.freeze:
            with torch.no_grad():
                out, _ = self.model(data)
        else:
            out, _ = self.model(data)
        # out is (N*L, 38, 17, 17)
        # COMPLETE CODE TO GET OUTPUT which should be (N) dimensional
        # linear, relu, linear
        out = out.reshape(N, -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        output = self.fc3(out)
                
        loss = None
        if targets is not None:
            loss = torch.nn.functional.mse_loss(output, targets, reduction='none')
            loss[:, 1:3] *= 2
            loss[:, 3:] *= 2.5
            loss = torch.mean(loss)
        return output, loss
    
class ResNetMC(torch.nn.Module):
    def __init__(self, num_outputs, H, W, hidden_size=256, num_heads=2, num_layers=2, freeze=True):
        super(ResNetMC, self).__init__()
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.num_outputs = num_outputs

        # Linear layers
        self.linear1 = nn.Linear(512, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.num_outputs)
        self.freeze = freeze

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
        if self.freeze:
            with torch.no_grad():
                x = self.backbone(x)
        else:
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
            loss = torch.nn.functional.mse_loss(output, targets, reduction='none')
            loss[:, 1:3] *= 2
            loss[:, 3:] *= 2.5
            loss = torch.mean(loss)
        print(output, loss)
        return output, loss


class SurveyModel(torch.nn.Module):
    """
    Base class for video models. Takes in video of (C x L x H x W) and outputs 
    a single vector of length D. Uses conv layers to extract features from
    video and then applies a linear layer to get output vector.
    Conv-> Relu -> Pool -> Concat -> Linear -> Relu -> Linear
    """
    def __init__(self, num_features: int, num_outputs: int, device='cpu'):
        super(SurveyModel, self).__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.l1 = nn.Linear(in_features=self.num_features, out_features=100)
        self.bn1 = nn.BatchNorm1d(num_features=100)
        self.d1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(in_features=100, out_features=100)
        self.bn2 = nn.BatchNorm1d(num_features=100)
        self.d2 = nn.Dropout()
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(in_features=100, out_features=100)
        self.bn3 = nn.BatchNorm1d(num_features=100)
        self.d3 = nn.Dropout()
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(in_features=100, out_features=self.num_outputs)

    
    def forward(self, x: torch.Tensor,  targets: Any = None, median_freq_weights = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape N x D
        Returns:
            torch.Tensor: Output vector of length D, Loss
        """
        x = self.l1(x)
        x = self.bn1(x)
        x = self.d1(x)
        x = self.relu1(x)
        
        x = self.l2(x)
        x = self.bn2(x)
        x = self.d2(x)
        x = self.relu2(x)
        
        x = self.l3(x)
        x = self.bn3(x)
        x = self.d3(x)
        x = self.relu3(x)

        x = self.l4(x)
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
            final_output = torch.nn.functional.softmax(output, dim=1)
        return final_output, loss