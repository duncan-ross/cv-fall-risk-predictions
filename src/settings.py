import os

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
BODY_MODEL_PATH = os.path.join(ABS_PATH, 'model/body_pose_model.pth')