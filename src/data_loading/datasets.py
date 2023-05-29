import sys
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import os
from typing import Any, List
import data_loading.transforms as transforms

# import transforms as transforms
import concurrent.futures
import PIL
import torchvision
import os
from settings import ABS_PATH, TRAIN_VIDEO_LENGTH_AVG, TRAIN_VIDEO_LENGTH_STD


class VideoLabelDataset(Dataset):
    def __init__(
        self,
        tabular_csv: str,
        video_folder: str,
        labels: List[str],
        transform: Any = None,
        video_transformer=transforms.VideoFilePathToTensor(),
        preload_videos: bool = True,
    ):
        """
        Args:
            tabular_csv (string): Path to the csv file with annotations.
            video_folder (string): Directory with all the videos.
            labels (List[str]): List of labels to be extracted from the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            video_transformer (callable, optional): Optional transform to be applied
                on a video.
            preload_videos (bool): Whether to preload all videos into memory.
        """
        self.video_folder = os.path.join(ABS_PATH, video_folder) 
        self.transform = transform
        self.video_transformer = video_transformer
        self.preload_videos = preload_videos

        df = pd.read_csv(os.path.join(ABS_PATH, tabular_csv))
        self.labels = df[labels].values
        self.ids = df["subjectid"].values
        # load videos into memory if desired
        self.videos = None
        if preload_videos:
            self.load_videos(video_transformer)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (subject id, video, label) where label is an array of labels.
        """
        if self.preload_videos:
            video = self.videos[index]
        else:
            video = self.video_transformer(
                os.path.join(self.video_folder, self.ids[index] + ".mp4")
            )

        label = self.labels[index]
        if self.transform:
            video = self.transform(video)
        return self.ids[index], video, torch.tensor(label)

    def load_videos(self, video_transformer: transforms.VideoFilePathToTensor):
        """
        Loads all videos into memory.
        """

        def process_video(id, video_folder):
            print(id)
            return video_transformer(os.path.join(video_folder, id + ".mp4"))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() // 2
        ) as executor:
            video_folder = self.video_folder
            self.videos = list(
                executor.map(
                    lambda id: video_transformer(
                        os.path.join(video_folder, id + ".mp4")
                    ),
                    self.ids,
                )
            )


class MotionCaptureDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        labels: List[str],
        transform: Any = None,
        video_transformer=transforms.VideoFilePathToTensor(fps=50, padding_mode="last"),
        preload_videos: bool = True,
    ):
        """
        Args:
            video_folder (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            video_transformer (callable, optional): Optional transform to be applied
                on a video.
            preload_videos (bool): Whether to preload all videos into memory.
        """
        # Iterate through the files in the root directory
        file_names = []
        for file_name in os.listdir(os.path.join(ABS_PATH, video_folder)):
            file_name_without_ext, _ = os.path.splitext(file_name)
            if (file_name_without_ext not in file_names) and (
                file_name_without_ext.startswith("s")
            ):
                file_names.append(file_name_without_ext)

        self.ids = file_names
        self.video_folder = os.path.join(ABS_PATH, video_folder)
        self.transform = transform
        self.video_transformer = video_transformer
        self.preload_videos = preload_videos  # Not possible atm.
        self.labels = labels
        self.videos = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (subject id, video, label) where label is an array of labels.
        """

        df = pd.read_csv(os.path.join(self.video_folder, self.ids[index] + ".csv"))
        tabular = torch.tensor(df[self.labels].values)
        n, m = tabular.shape
        # reduce the number of rows to be equivalent with the video
        denom = 100 // self.video_transformer.fps
        reduced_rows = n // denom
        tabular = (
            tabular[: reduced_rows * denom, :].view(reduced_rows, denom, m).mean(dim=1)
        )
        # tabular = torch.abs(tabular)

        if self.preload_videos:
            video = self.videos[index]
        else:
            # TODO can we get away with not redefining this every time??
            self.video_transformer.max_len = tabular.shape[0]
            video = self.video_transformer(
                os.path.join(self.video_folder, self.ids[index] + ".mp4")
            )

        if self.transform:
            video = self.transform(video)

        try:
            assert tabular.shape[0] == video.shape[1]
        except AssertionError:
            print("Assertion Error: Number of rows mismatch!")
            print("Number of rows in DataFrame:", tabular.shape[0])
            print("Number of elements in 'video':", video.shape[1])
            sys.exit(0)

        return self.ids[index], video, tabular

    def load_videos(self, video_transformer: transforms.VideoFilePathToTensor):
        """
        Loads all videos into memory.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() // 2
        ) as executor:
            video_folder = self.video_folder
            self.videos = list(
                executor.map(
                    lambda id: video_transformer(
                        os.path.join(video_folder, id + ".mp4")
                    ),
                    self.ids,
                )
            )


class SurveyDataset(Dataset):
    def __init__(
        self,
        tabular_csv: str,
        labels: List[str],
    ):
        """
        Args:
            tabular_csv (string): Path to the csv file with annotations.
            labels (List[str]): List of labels to be extracted from the csv file.
        """

        df = pd.read_csv(os.path.join(ABS_PATH, tabular_csv))
        self.labels = df[labels].values
        self.ids = df["subjectid"].values
        # load videos into memory if desired
        self.data = df.drop(columns=["y_fall_risk", "y_fall_risk_binary", "subjectid"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (subject id, video, label) where label is an array of labels.
        """
        obs = self.data.iloc[index]
        label = self.labels[index]
        return self.ids[index], torch.tensor(obs), torch.tensor(label)


class FusionDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        tabular_csv: str,
        tabular_train_csv: str,
        labels: List[str],
        transform: Any = None,
        video_transformer=transforms.VideoFilePathToTensor(fps=50, padding_mode="last"),
        preload_videos: bool = True,
    ):
        """
        Args:
            video_folder (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            video_transformer (callable, optional): Optional transform to be applied
                on a video.
            preload_videos (bool): Whether to preload all videos into memory.
        """
        # Iterate through the files in the root directory
        file_names = []
        for file_name in os.listdir(os.path.join(ABS_PATH, video_folder)):
            file_name_without_ext, _ = os.path.splitext(file_name)
            if (file_name_without_ext not in file_names):
                file_names.append(file_name_without_ext)

        self.ids = file_names
        self.video_folder = os.path.join(ABS_PATH, video_folder)
        self.transform = transform
        self.video_transformer = video_transformer

        # load videos into memory if desired
        self.preload_videos = preload_videos  # Not possible atm.
        self.videos = None

        df = pd.read_csv(os.path.join(ABS_PATH, tabular_csv))
        self.labels = df[labels].values
        #self.ids = df["subjectid"].values # This SHOULD be the same as the file_names since we need them aligned
        # dropping extra features that have no standard deviation
        self.data = df.drop(columns=["y_fall_risk", "y_fall_risk_binary", "subjectid", "SC1", "WHISH"])
        # standardize data based on training data
        df_train = pd.read_csv(os.path.join(ABS_PATH, tabular_train_csv)).drop(columns=["y_fall_risk", "y_fall_risk_binary", "subjectid", "SC1", "WHISH"])
        self.data = (self.data - df_train.mean()) / df_train.std()


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (subject id, video, label) where label is an array of video_labels.
        """

        if self.preload_videos:
            video = self.videos[index]
        else:
            # TODO can we get away with not redefining this every time??
            video = self.video_transformer(
                os.path.join(self.video_folder, self.ids[index] + ".mp4")
            )

        if self.transform:
            video = self.transform(video)

        # append length of video to tabular data
        video_len = video.shape[1]/self.video_transformer.fps
        # standardize video length based on training data
        video_len = (video_len - TRAIN_VIDEO_LENGTH_AVG) / TRAIN_VIDEO_LENGTH_STD
        # 
        tabular = torch.tensor(self.data.iloc[index], dtype=torch.float32)
        tabular = torch.cat((tabular, torch.tensor([video_len])))
        output_label = torch.tensor(self.labels[index])
        return self.ids[index], (video, tabular), output_label

    def load_videos(self, video_transformer: transforms.VideoFilePathToTensor):
        """
        Loads all videos into memory.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() // 2
        ) as executor:
            video_folder = self.video_folder
            self.videos = list(
                executor.map(
                    lambda id: video_transformer(
                        os.path.join(video_folder, id + ".mp4")
                    ),
                    self.ids,
                )
            )


if __name__ == "__main__":
    # test for VideoDataset
    video_transformer = transforms.VideoFilePathToTensor(
        max_len=None, fps=5, padding_mode="last"
    )
    dataset = VideoLabelDataset(
        tabular_csv="data/processed/train-survey-data.csv",
        video_folder="data/processed/train-videos",
        labels=["y_fall_risk"],
        video_transformer=video_transformer,
        transform=torchvision.transforms.Compose(
            [
                # transforms.VideoRandomCrop([512, 512]),
                # transforms.VideoResize([256, 256]),
            ]
        ),
        preload_videos=False,
    )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    vid_legths = []
    for i, data in enumerate(test_loader):
        subj_id, videos, labels = data
        # keep running mean of video length
        video_len = videos.shape[2] / video_transformer.fps
        vid_legths.append(video_len)
        print(subj_id, videos.size(), labels, video_len)
    print("mean", np.mean(vid_legths))
    print("std", np.std(vid_legths))


if __name__ == "__main__SKIP":
    # test for VideoDataset
    video_transformer = transforms.VideoFilePathToTensor(
        max_len=None, fps=1, padding_mode="last"
    )
    dataset = MotionCaptureDataset(
        video_folder="data/motion_capture",
        labels=["pelvis_tilt", "ankle_angle_l"],
        video_transformer=video_transformer,
        transform=torchvision.transforms.Compose([]),
        preload_videos=False,
    )
    subj_id, video, label = dataset[0]
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for subj_id, videos, labels in test_loader:
        # print(subj_id, videos.size(), label)
        print(subj_id, videos.size(), labels.size())
