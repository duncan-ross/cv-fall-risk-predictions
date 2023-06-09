# cv-fall-risk-predictions

Note that we get the de-identified videos from a [Google Drive folder](https://drive.google.com/drive/folders/1C6777-AFWU2LUPx9ydqtuJjqrMi2oDwS) and store them in `data/raw/videos` to work with the pipeline

Download `body_pose_model.pth` and `hand_pose_model.pth` if you want to use the PyTorch implementation of OpenPose. These can be downloaded from https://github.com/Hzzone/pytorch-openpose/

For motion capture:

You will need to download our motion capture data from the google drive:
https://drive.google.com/drive/folders/16DcuIVlI7cR5-f9jCWDW5SkYiO2YCVOp?usp=sharing

and place into `data/motion_capture`.

Then run the `02-preprocess-data.py` file to generate the CSV corresponding to each video.
