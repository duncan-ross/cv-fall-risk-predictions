import os
import shutil


"""
Usage: python3 move_videos.py

Download videos from drive, unzip, and move them to a folder called videos in the data directory.
"""

# Specify the paths to the videos folder and the text files
videos_folder = "./data/videos"
train_file = "./data/processed/train-video-ids.txt"
val_file =  "./data/processed/val-video-ids.txt"
test_file =  "./data/processed/test-video-ids.txt"

# Create the train, val, and test subfolders if they don't exist already
train_folder = os.path.join(videos_folder, 'train')
val_folder = os.path.join(videos_folder, 'val')
test_folder = os.path.join(videos_folder, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Function to move videos to the specified destination folder
def move_video(video_name, destination_folder):
    source_path = os.path.join(videos_folder, video_name + '.mp4')
    destination_path = os.path.join(destination_folder, video_name + '.mp4')
    shutil.move(source_path, destination_path)

# Read the video names from the train video ids file and move them to the train folder
with open(train_file, 'r') as train_ids:
    train_videos = train_ids.read().splitlines()
    for video_name in train_videos:
        move_video(video_name, train_folder)

# Read the video names from the val video ids file and move them to the val folder
with open(val_file, 'r') as val_ids:
    val_videos = val_ids.read().splitlines()
    for video_name in val_videos:
        move_video(video_name, val_folder)

# Read the video names from the test video ids file and move them to the test folder
with open(test_file, 'r') as test_ids:
    test_videos = test_ids.read().splitlines()
    for video_name in test_videos:
        move_video(video_name, test_folder)
