import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import tempfile
import streamlit as st
import RRDBNet_arch as arch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/RRDB_ESRGAN_x4.pth'  # Pre-trained model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model.to(device)

model = load_model()

# Functions for processing
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
    return idx

def super_resolve_frames(input_dir, output_dir, progress_bar):
    frame_paths = sorted(glob.glob(osp.join(input_dir, '*.png')))
    total_frames = len(frame_paths)

    for idx, path in enumerate(frame_paths):
        base = osp.splitext(osp.basename(path))[0]

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
        output_path = osp.join(output_dir, f'{base}_sr.png')
        cv2.imwrite(output_path, output)

        # Update progress bar
        progress_bar.progress((idx + 1) / total_frames, text=f"Processing {idx + 1}/{total_frames} frames...")

def frames_to_video(input_dir, output_video_path, fps):
    frame_paths = sorted(glob.glob(osp.join(input_dir, '*.png')))
    if not frame_paths:
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

# Streamlit App
st.title("Video Super-Resolution")
st.write("Upload a video to enhance its resolution using ESRGAN.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded video
        input_video_path = osp.join(temp_dir, uploaded_file.name)
        with open(input_video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Temporary paths
        temp_frame_dir = osp.join(temp_dir, "frames")
        os.makedirs(temp_frame_dir, exist_ok=True)
        output_video_path = osp.join(temp_dir, "output_video.mp4")

        # Extract frames
        st.write("Extracting frames from video...")
        frame_count = extract_frames(input_video_path, temp_frame_dir)
        st.write(f"Extracted {frame_count} frames.")

        # Super-resolution processing with progress bar
        st.write("Applying super-resolution to frames...")
        progress_bar = st.progress(0)
        super_resolve_frames(temp_frame_dir, temp_frame_dir, progress_bar)

        # Reassemble frames
        st.write("Reassembling frames into video...")
        original_fps = cv2.VideoCapture(input_video_path).get(cv2.CAP_PROP_FPS)
        frames_to_video(temp_frame_dir, output_video_path, fps=original_fps)

        # Display success message
        st.success("Video super-resolution completed successfully!")

        # Provide download link
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Enhanced Video",
                data=f,
                file_name="enhanced_video.mp4",
                mime="video/mp4"
            )
