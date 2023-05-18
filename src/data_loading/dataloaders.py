from typing import Tuple, Any, List
import torch
from torch.utils.data import DataLoader
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import warnings
import numpy as np
import torchvision

from data_loading.datasets import VideoLabelDataset

def get_vid_data_loaders(video_transformer: Any, batch_size: int = 32, val_batch_size: int = 16, test_batch_size: int = 16, 
transforms: Any = None, preload_videos: bool = False, labels: List = ['y_fall_risk'], num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}
    for ds in ['train', 'val', 'test']:
        # just do the resize for the val/test set
        transforms = transforms if ds == 'train' else torchvision.transforms.Compose([
                transforms.transforms[0]
            ])
        dataset = VideoLabelDataset(
            tabular_csv=f'data/processed/{ds}-survey-data.csv', 
            video_folder=f'data/processed/{ds}-videos', 
            labels=labels, 
            video_transformer=video_transformer,
            transform=transforms,
            preload_videos=preload_videos
        )
        ds_dict[ds] = DataLoader(dataset, 
        batch_size=batch_size if ds == 'train' else val_batch_size if ds == 'val' else test_batch_size, 
        shuffle=True, num_workers=num_workers)

    
    return ds_dict['train'], ds_dict['val'], ds_dict['test']


def get_mc_data_loaders(video_transformer: Any, batch_size: int = 32, val_batch_size: int = 16, test_batch_size: int = 16, 
transforms: Any = None, preload_videos: bool = False, labels: List = ['y_fall_risk'], num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}

    for ds in ['train', 'val']:
        transforms = transforms if ds == 'train' else torchvision.transforms.Compose([
                transforms.transforms[0]
            ])
        dataset = MotionCaptureDataset(
            video_folder=f'data/motion_capture', 
            labels=labels, 
            video_transformer=video_transformer,
            transform=transforms,
            preload_videos=preload_videos
        )
        ds_dict['ds'] = DataLoader(dataset, 
        batch_size=batch_size if ds == 'train' else val_batch_size if ds == 'val' else test_batch_size, 
        shuffle=True, num_workers=num_workers)

    return ds_dict['train'], ds_dict['val'], None