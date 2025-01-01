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
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)  # Avoid future warnings
    model.eval()
    return model.to(device)

model = load_model()

# Functions for processing
def process_image(image, model):
    # Preprocess image
    img = image * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Apply super-resolution
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output

# Streamlit App
st.title("Image Super-Resolution")
st.write("Upload an image to enhance its resolution using ESRGAN.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read image
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    
    # Display original image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Apply super-resolution
    st.write("Enhancing image resolution...")
    result_image = process_image(image, model)
    
    # Display result
    st.image(result_image, caption="Enhanced Image", use_container_width=True)
    
    # Provide download link
    _, ext = osp.splitext(uploaded_file.name)
    output_image_path = tempfile.mktemp(suffix=f"_enhanced{ext}")
    cv2.imwrite(output_image_path, result_image)

    with open(output_image_path, "rb") as f:
        st.download_button(
            label="Download Enhanced Image",
            data=f,
            file_name="enhanced_image.png",
            mime="image/png"
        )
