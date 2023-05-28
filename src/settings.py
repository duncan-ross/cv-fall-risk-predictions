import os

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
BODY_MODEL_PATH = os.path.join(ABS_PATH, 'model/body_pose_model.pth')

MC_RESPONSES = [
            "pelvis_tilt",
            "ankle_angle_l",
            "ankle_angle_r",
            "hip_adduction_r",
            "hip_adduction_l",
        ]

TRAIN_VIDEO_LENGTH_AVG=13.014925373134329
TRAIN_VIDEO_LENGTH_STD=3.4991791673444075