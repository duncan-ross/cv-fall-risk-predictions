import sys
import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd
import os
from typing import Any, List
#import data_loading.transforms as transforms
import transforms as transforms
import concurrent.futures
import PIL
import torchvision

class VideoLabelDataset(Dataset):
    def __init__(self, tabular_csv: str, video_folder: str, labels: List[str], transform:Any =None, 
    video_transformer=transforms.VideoFilePathToTensor(), preload_videos: bool=True):
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
        self.video_folder = video_folder
        self.transform = transform
        self.video_transformer = video_transformer
        self.preload_videos = preload_videos

        df = pd.read_csv(tabular_csv)
        self.labels = df[labels].values
        self.ids = df['subjectid'].values
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
            video =  self.video_transformer(os.path.join(self.video_folder, self.ids[index] + '.mp4'))
        
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
            return video_transformer(os.path.join(video_folder, id + '.mp4'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            video_folder = self.video_folder
            self.videos = list(executor.map(
                lambda id: video_transformer(os.path.join(video_folder, id + '.mp4')),
                self.ids
            ))

class MotionCaptureDataset(Dataset):
    def __init__(self, video_folder: str, labels: List[str], transform:Any =None, 
    video_transformer=transforms.VideoFilePathToTensor(fps=50, padding_mode='last'), preload_videos: bool=True):
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
        for file_name in os.listdir(video_folder):
            file_name_without_ext, _ = os.path.splitext(file_name)
            if file_name_without_ext not in file_names:
                file_names.append(file_name_without_ext)
        
        self.ids = file_names
        self.video_folder = video_folder
        self.transform = transform
        self.video_transformer = video_transformer
        self.preload_videos = preload_videos # Not possible atm.
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

        df = pd.read_csv(os.path.join(self.video_folder, self.ids[index] + '.csv'))
        tabular = torch.tensor(df[self.labels].values)
        n, m = tabular.shape
        reduced_rows = n // 2
        tabular = tabular[:reduced_rows*2, :].view(reduced_rows, 2, m).mean(dim=1)

        if self.preload_videos:
            video = self.videos[index]
        else:
            # TODO can we get away with not redefining this every time??
            self.video_transformer.max_len = df.shape[0]
            video =  video_transformer(os.path.join(self.video_folder, self.ids[index] + '.mp4'))
        
        if self.transform:
            video = self.transform(video)

        try:
            assert df.shape[0] == video.shape[1]
        except AssertionError:
            print("Assertion Error: Number of rows mismatch!")
            print("Number of rows in DataFrame:", df.shape[0])
            print("Number of elements in 'video':", video.shape[1])
            sys.exit(0)

        return self.ids[index], video, tabular
    
    def load_videos(self, video_transformer: transforms.VideoFilePathToTensor):
        """
        Loads all videos into memory.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
            video_folder = self.video_folder
            self.videos = list(executor.map(
                lambda id: video_transformer(os.path.join(video_folder, id + '.mp4')),
                self.ids
            ))
    

if __name__ == '__main__SKIP':
    # test for VideoDataset
    video_transformer = transforms.VideoFilePathToTensor(max_len=None, fps=50, padding_mode='last')
    dataset = VideoLabelDataset(
        tabular_csv='data/processed/val-survey-data.csv', 
        video_folder='data/processed/val-videos', 
        labels=['y_fall_risk'], 
        video_transformer=video_transformer,
        transform=torchvision.transforms.Compose([
            # transforms.VideoRandomCrop([512, 512]),
            # transforms.VideoResize([256, 256]),
        ]),
        preload_videos=False
    )
    subj_id, video, label = dataset[0]
    print(subj_id, video.size(), label)
    frame1 = torchvision.transforms.ToPILImage()(video[:, 29, :, :])
    frame2 = torchvision.transforms.ToPILImage()(video[:, 19, :, :])
    frame1.show()
    frame2.show()

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    for subj_id, videos, labels in test_loader:
        print(subj_id, videos.size(), label)

if __name__ == '__main__':
    # test for VideoDataset
    video_transformer = transforms.VideoFilePathToTensor(max_len=None, fps=50, padding_mode='last')
    dataset = MotionCaptureDataset(
        video_folder='data/motion_capture', 
        labels=['pelvis_tilt', 'ankle_angle_l'],
        video_transformer=video_transformer,
        transform=torchvision.transforms.Compose([]),
        preload_videos=False
    )
    subj_id, video, label = dataset[0]
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for subj_id, videos, labels in test_loader:
        #print(subj_id, videos.size(), label)
        continue