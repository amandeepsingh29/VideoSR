# Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) for Image and Video Super-Resolution

:sparkles: **New Updates**

Experience the cutting-edge of **Image and Video Super-Resolution** technology with ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks). This product offers state-of-the-art resolution enhancement for low-quality images and videos, delivering unparalleled clarity and detail restoration.

## Features
- **4x Image and Video Upscaling**: Transform low-resolution visuals into high-quality content.
- **Artifact Removal**: Eliminate compression artifacts for cleaner output.
- **Wide Format Support**: Works with PNG, JPEG, MP4, AVI, and other common formats.
- **User-Friendly Interface**: Simple upload and download workflow.
- **Multi-Channel Support**: Handles grayscale, RGB, and 16-bit images effortlessly.

## Model Overview
Our application utilizes a pretrained ESRGAN model with the following benefits:
1. **Unmatched Visual Quality**: Generates sharp textures and lifelike details.
2. **Advanced GAN Framework**: Uses Relativistic average GAN for superior perceptual quality.
3. **Optimized Loss Functions**: Balances fidelity and perceptual quality for optimal results.

## Quick Start Guide
### Prerequisites
- Python 3.8 or later
- [PyTorch](https://pytorch.org/) >= 1.0
- Install additional dependencies:
  ```
  pip install numpy opencv-python moviepy
  ```

### Steps to Use
1. Clone the repository:
   ```
   git clone https://github.com/your-repo/ImageSuperResolution.git
   cd ImageSuperResolution
   ```
2. Download the pretrained ESRGAN model:
   - [Pretrained Model](https://drive.google.com/drive/folder/path-to-model) and save it in the `models/` directory.
3. For image upscaling:
   - Add your low-resolution images to the `input/images/` folder.
   - Run the upscaling script:
     ```
     python upscale_images.py
     ```
   - View the enhanced images in the `output/images/` folder.
4. For video upscaling:
   - Add your low-resolution videos to the `input/videos/` folder.
   - Run the video upscaling script:
     ```
     python upscale_videos.py
     ```
   - View the enhanced videos in the `output/videos/` folder.

## Example Workflow
### Image Upscaling
- **Original Image:**
  
  ![Low-Resolution Image](assets/input_example.jpg)

- **Super-Resolved Image:**
  
  ![High-Resolution Image](assets/output_example.jpg)

### Video Upscaling
- **Original Video Frame:**
  
  ![Low-Resolution Frame](assets/input_video_frame.jpg)

- **Super-Resolved Video Frame:**
  
  ![High-Resolution Frame](assets/output_video_frame.jpg)

## Technical Details
**Key Features of ESRGAN Architecture:**
- **Residual-in-Residual Dense Blocks (RRDB):** Improves gradient flow and stability.
- **Batch Normalization Removal:** Reduces artifacts for clearer images.
- **Perceptual Loss:** Calculated using VGG features before activation for realistic textures.

## Applications
- **Photography**: Restore and enhance personal photo collections.
- **E-commerce**: Improve product visuals to boost sales.
- **Healthcare**: Upscale diagnostic images for better analysis.
- **Education and Research**: Enhance images for presentations and publications.
- **Video Production**: Restore old footage, enhance video quality for editing or streaming.

## FAQ
1. **What sets this apart from other upscaling methods?**
   - Unlike traditional methods, ESRGAN generates natural-looking details and textures by prioritizing perceptual quality.
2. **Is this model suitable for real-time use?**
   - While optimized for fast inference, hardware specifications will influence real-time performance.
3. **Can this upscale videos of any format?**
   - Yes, as long as the format is supported by FFmpeg or MoviePy libraries.

## Acknowledgements
This project builds on the open-source [ESRGAN](https://github.com/xinntao/ESRGAN) framework. A big thank you to the developers and contributors who made this possible.

### Citation
If you utilize this tool for research, kindly cite the original work:
```
@InProceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},n    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
```

Enhance your images and videos today with ESRGAN technology!

