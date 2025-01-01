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

# Load model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print(f'Model loaded from {model_path}. \nProcessing video...')

# Create temporary frame directory
os.makedirs(temp_frame_dir, exist_ok=True)

# Function to extract frames from video
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = osp.join(output_dir, f'frame_{idx:04d}.png')
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    print(f'Extracted {idx} frames from video.')

# Function to apply super-resolution to frames
def super_resolve_frames(input_dir, output_dir):
    frame_paths = sorted(glob.glob(osp.join(input_dir, '*.png')))
    os.makedirs(output_dir, exist_ok=True)
    for idx, path in enumerate(frame_paths):
        base = osp.splitext(osp.basename(path))[0]
        print(f'Processing frame {idx + 1}/{len(frame_paths)}: {base}')

        # Read frame
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)

        # Apply super-resolution
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        # Save the result
        cv2.imwrite(osp.join(output_dir, f'{base}_sr.png'), output)

# Function to reassemble frames into video
def frames_to_video(input_dir, output_video_path, fps):
    frame_paths = sorted(glob.glob(osp.join(input_dir, '*.png')))
    if not frame_paths:
        print("No frames found for video reconstruction.")
        return
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    video_writer.release()
    print(f'Video saved to {output_video_path}')

# Main processing pipeline
if __name__ == '__main__':
    # Step 1: Extract frames from video
    extract_frames(input_video_path, temp_frame_dir)

    # Step 2: Apply super-resolution to frames
    super_resolve_frames(temp_frame_dir, temp_frame_dir)

    # Step 3: Reassemble frames into video
    original_fps = cv2.VideoCapture(input_video_path).get(cv2.CAP_PROP_FPS)
    frames_to_video(temp_frame_dir, output_video_path, fps=original_fps)

    # Clean up temporary frames (optional)
    for temp_file in glob.glob(osp.join(temp_frame_dir, '*.png')):
        os.remove(temp_file)
    os.rmdir(temp_frame_dir)

    print("Video super-resolution completed successfully.")
