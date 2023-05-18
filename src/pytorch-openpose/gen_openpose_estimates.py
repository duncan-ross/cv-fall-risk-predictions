import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from src import model
from src import util
from src.body import Body
from src.hand import Hand
import multiprocessing as mp
import pandas as pd

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)



def process_video(video_file):
    # load models
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    cap = cv2.VideoCapture(video_file)

    # get video file info
    ffprobe_result = ffprobe(video_file)
    info = json.loads(ffprobe_result.json)
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    input_fps = videoinfo["avg_frame_rate"]
    # input_fps = float(input_fps[0])/float(input_fps[1])
    input_pix_fmt = videoinfo["pix_fmt"]
    input_vcodec = videoinfo["codec_name"]


    # collect df of all frames
    df_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        candidate, _ = body_estimation(frame)
        if len(candidate) == 0:
            continue
        cand_df = pd.DataFrame(candidate, columns=['x', 'y', 'score', 'id'])
        cand_df['frame'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
        df_list.append(cand_df)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # combine all frames
    df = pd.concat(df_list)
    df['subj_id'] = video_file.split('/')[-1].replace('.mp4', '')

    cap.release()
    cv2.destroyAllWindows()
    return df


if __name__ == '__main__':
    # open specified video
    parser = argparse.ArgumentParser(
            description="Process a video annotating poses detected.")
    parser.add_argument('--directory', type=str, help='Video directory to process', default='data/processed/train-videos/')
    args = parser.parse_args()

    # get all video files in directory
    video_files = glob(os.path.join(args.directory, '*.mp4'))

    # process each video file
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result_list = pool.map(process_video, video_files)
    df = pd.concat(result_list)
    df.to_csv(os.path.join(args.directory, 'all.csv'), index=False)
    