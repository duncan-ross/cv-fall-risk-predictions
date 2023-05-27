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

from data_loading.datasets import *


def get_vid_data_loaders(
    video_transformer: Any,
    batch_size: int = 32,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    transforms: Any = None,
    preload_videos: bool = False,
    labels: List = ["y_fall_risk"],
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}
    for ds in ["train", "val", "test"]:
        # just do the resize for the val/test set
        if len(transforms.transforms) > 0:
            transforms = (
                transforms
                if ds == "train"
                else torchvision.transforms.Compose([transforms.transforms[0]])
            )
        else:
            transforms = transforms

        dataset = VideoLabelDataset(
            tabular_csv=f"data/processed/{ds}-survey-data.csv",
            video_folder=f"data/processed/{ds}-videos",
            labels=labels,
            video_transformer=video_transformer,
            transform=transforms,
            preload_videos=preload_videos,
        )
        ds_dict[ds] = DataLoader(
            dataset,
            batch_size=batch_size
            if ds == "train"
            else val_batch_size
            if ds == "val"
            else test_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return ds_dict["train"], ds_dict["val"], ds_dict["test"]


def get_mc_data_loaders(
    video_transformer: Any,
    batch_size: int = 32,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    transforms: Any = None,
    preload_videos: bool = False,
    labels: List = [
        "pelvis_tilt",
        "ankle_angle_l",
        "ankle_angle_r",
        "hip_adduction_r",
        "hip_adduction_l",
    ],
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}

    for ds in ["train", "val", "test"]:
        transforms = (
            transforms
            if ds == "train"
            else torchvision.transforms.Compose([transforms.transforms[0]])
        )
        dataset = MotionCaptureDataset(
            video_folder=f"data/motion_capture/{ds}",
            labels=labels,
            video_transformer=video_transformer,
            transform=transforms,
            preload_videos=preload_videos,
        )
        ds_dict[ds] = DataLoader(
            dataset,
            batch_size=batch_size
            if ds == "train"
            else val_batch_size
            if ds == "val"
            else test_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return ds_dict["train"], ds_dict["val"], ds_dict["test"]


def get_survey_data_loaders(
    batch_size: int = 32,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    transforms: Any = None,
    labels: List = ["y_fall_risk"],
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}
    for ds in ["train", "val", "test"]:
        # just do the resize for the val/test set
        if len(transforms.transforms) > 0:
            transforms = (
                transforms
                if ds == "train"
                else torchvision.transforms.Compose([transforms.transforms[0]])
            )
        else:
            transforms = transforms

        dataset = SurveyDataset(
            tabular_csv=f"data/processed/{ds}-survey-data.csv",
            labels=labels,
        )
        ds_dict[ds] = DataLoader(
            dataset,
            batch_size=batch_size
            if ds == "train"
            else val_batch_size
            if ds == "val"
            else test_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return ds_dict["train"], ds_dict["val"], ds_dict["test"]


def get_fusion_data_loaders(
    video_transformer: Any,
    batch_size: int = 32,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    transforms: Any = None,
    preload_videos: bool = False,
    labels: List = ["y_fall_risk"],
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_dict = {}

    for ds in ["train", "val", "test"]:
        transforms = (
            transforms
            if ds == "train"
            else torchvision.transforms.Compose([transforms.transforms[0]])
        )
        dataset = FusionDataset(
            video_folder=f"data/videos/{ds}",
            tabular_csv=f"data/processed/{ds}-survey-data.csv",
            tabular_train_csv=f"data/processed/train-survey-data.csv",
            labels=labels,
            video_transformer=video_transformer,
            transform=transforms,
            preload_videos=preload_videos,
        )
        ds_dict[ds] = DataLoader(
            dataset,
            batch_size=batch_size
            if ds == "train"
            else val_batch_size
            if ds == "val"
            else test_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=fusion_collate_fn,
        )

    return ds_dict["train"], ds_dict["val"], ds_dict["test"]


def collate_fn(batch):
    # handle videos of different lengths
    # batch is a list of tuples (subj_id, video, label)
    subj_id, video, label = zip(*batch)
    video = torch.cat(video[0], dim=1)
    label = torch.cat(label, dim=0)
    return subj_id, video, label

def fusion_collate_fn(batch):
    subj_id, data, label = zip(*batch)
    video, tabular = data[0]    
    video = torch.cat(tuple(video[0]), dim=1)
    return subj_id, (video, tabular), label
