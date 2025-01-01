from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import torch
import tempfile
from RRDBNet_arch import RRDBNet

app = Flask(__name__)

# Load ESRGAN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_path = 'models/RRDB_ESRGAN_x4.pth'
    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model.to(device)

model = load_model()

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return idx

def super_resolve_frames(input_dir, output_dir):
    frame_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    for idx, path in enumerate(frame_paths):
        base = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        output_path = os.path.join(output_dir, f'{base}_sr.png')
        cv2.imwrite(output_path, output)

def frames_to_video(input_dir, output_video_path, fps):
    frame_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    if not frame_paths:
        return
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    video_writer.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['file']
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(input_video_path)

            temp_frame_dir = os.path.join(temp_dir, 'frames')
            os.makedirs(temp_frame_dir, exist_ok=True)
            output_video_path = os.path.join(temp_dir, 'output_video.mp4')

            extract_frames(input_video_path, temp_frame_dir)
            super_resolve_frames(temp_frame_dir, temp_frame_dir)

            original_fps = cv2.VideoCapture(input_video_path).get(cv2.CAP_PROP_FPS)
            frames_to_video(temp_frame_dir, output_video_path, original_fps)

            return send_file(output_video_path, as_attachment=True, download_name='enhanced_video.mp4')

if __name__ == '__main__':
    app.run(debug=True)
