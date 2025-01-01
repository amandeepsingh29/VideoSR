document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const videoInput = document.getElementById('video-input');
    const progressContainer = document.getElementById('progress-container');
    const downloadContainer = document.getElementById('download-container');
    const progressBar = document.getElementById('progress-bar');
    const downloadLink = document.getElementById('download-link');
    
    if (!videoInput.files.length) {
        alert('Please upload a video file.');
        return;
    }

    const formData = new FormData();
    formData.append('video', videoInput.files[0]);

    progressContainer.style.display = 'block';

    try {
        const response = await fetch('/process_video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to process the video.');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        downloadContainer.style.display = 'block';
        downloadLink.href = url;

        progressContainer.style.display = 'none';
    } catch (error) {
        alert(error.message);
        progressContainer.style.display = 'none';
    }
});
