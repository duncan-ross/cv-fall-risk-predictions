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
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        # apply resnet backbone but do not change weights
        with torch.no_grad():
            x = self.backbone(x)
        x = x.reshape(N, L, -1)
        x, hidden = self.lstm(x)
        x = hidden[0].transpose(0, 1).reshape(N, -1)
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        output = self.layer2(x)

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
                final_output[:, 0] = torch.sigmoid(output[:, 0])
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
                norm=nn.LayerNorm(512),
                batch_first=True
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
                    # binary cross entropy with class weights torch logits
                    loss = torch.nn.BCEWithLogitsLoss(weight=median_freq_weights)(output, targets)
            # softmax but do not do gradient
            if targets.shape[1] == 1:
                with torch.no_grad():
                    final_output = torch.nn.functional.softmax(output, dim=1)
            return final_output, loss


class BaseOpenPose(torch.nn.Module):
        def __init__(self, num_outputs, L, H, W, hidden_size=512, num_heads=8, num_layers=6, device='cpu'):
            super(BaseTransformer, self).__init__()
            self.device = device
            self.model = bodypose_model()
            model_dict = util.transfer(self.model, torch.load('model/body_pose_model.pth'))
            self.model.load_state_dict(model_dict)
            self.model = self.model.to(device)
            modules = list(self.model.children())[:-1]
            self.backbone = torch.nn.Sequential(*modules)

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
            data = process_image(out).to(self.device)

            out, _ = self.model(data)
            # out is (N*L, 38, 17, 17)
            # COMPLETE CODE TO GET OUTPUT which should be (N) dimensional
            # linear, relu, linear
            out = out.reshape(N, L, -1)
            output, hid = self.rnn(out)

            hid = hid.reshape(N, -1)
            output = self.fc1(hid)
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
                final_output = torch.nn.functional.softmax(output, dim=1)
            print(final_output)
            return final_output, loss

# Helper function to proc image for openpose
def process_image(x):
    # Convert Torch tensor to numpy array
    numpy_images = (x*255).numpy()

    # Transpose the image array if necessary
    if len(numpy_images.shape) == 4:
        numpy_images = np.transpose(numpy_images, (0, 2, 3, 1))
    else:
        numpy_images = np.transpose(numpy_images, (1, 2, 0))

    oriImgs = []
    for numpy_image in numpy_images:
        oriImg = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        oriImgs.append(oriImg)
    oriImgs = np.array(oriImgs)

    scale_search = 0.5
    boxsize = 368
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05

    # Calculate the multiplier for scaling
    multipliers = scale_search * boxsize / oriImgs.shape[1]

    # Resize images
    resized_images = []
    for oriImg in oriImgs:
        resized_img = cv2.resize(oriImg, (0, 0), fx=multipliers, fy=multipliers, interpolation=cv2.INTER_CUBIC)
        resized_images.append(resized_img)
    resized_images = np.array(resized_images)

    # Pad images
    padded_images = []
    pads = []
    for i in range(x.shape[0]):
        padded_image, pad = util.padRightDownCorner(resized_images[i], stride, padValue)
        padded_images.append(padded_image)
        pads.append(pad)
    padded_images = np.stack(padded_images)
    pads = np.stack(pads)

    # Transpose and normalize images
    im = np.transpose(np.float32(padded_images), (0, 3, 1, 2)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    return torch.from_numpy(im.astype(np.float32))
    