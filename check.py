import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Paths
model_path = 'models/RRDB_ESRGAN_x4.pth'  # Pre-trained model
input_video_path = 'input_video.mp4'  # Path to input video
output_video_path = 'output_video.mp4'  # Path to save the output video
temp_frame_dir = 'temp_frames'  # Temporary directory to store frames

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')  # Check whether it's running on GPU or CPU