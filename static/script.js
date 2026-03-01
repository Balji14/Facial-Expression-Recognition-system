const video = document.getElementById('webcam');
const canvas = document.getElementById('photo-canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('capture-btn');
const resetBtn = document.getElementById('reset-btn');
const statusOverlay = document.getElementById('status-overlay');
const resultContainer = document.getElementById('result-container');
const emotionLabel = document.getElementById('emotion-label');
const confidenceLabel = document.getElementById('confidence-label');
const errorMsg = document.getElementById('error-message');

let stream = null;

// Initialize WebCam
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            statusOverlay.classList.add('hidden');
            captureBtn.disabled = false;
        };
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        statusOverlay.innerHTML = "Webcam access denied or unavailable.<br>Please allow camera permissions.";
        statusOverlay.style.background = "rgba(220, 53, 69, 0.8)";
    }
}

// Stop WebCam
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

// Draw the current video frame onto the canvas
function snapPhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Show the freeze frame canvas, hide live video
    canvas.style.display = 'block';
    video.style.display = 'none';
}

// Send Image to Backend API
async function predictEmotion(imageDataUrl) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageDataUrl })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Server error occurred.");
        }

        return result;
    } catch (err) {
        throw err;
    }
}

// Button Click Event
captureBtn.addEventListener('click', async () => {
    // UI Loading state
    captureBtn.classList.add('loading');
    captureBtn.disabled = true;
    errorMsg.classList.add('hidden');
    resultContainer.classList.add('hidden');

    try {
        snapPhoto();
        // Optionally downscale the image on the client to reduce payload
        const maxDim = 480;
        let result;
        if (canvas.width > maxDim || canvas.height > maxDim) {
            const tmp = document.createElement('canvas');
            const scale = Math.min(maxDim / canvas.width, maxDim / canvas.height);
            tmp.width = canvas.width * scale;
            tmp.height = canvas.height * scale;
            tmp.getContext('2d').drawImage(canvas, 0, 0, tmp.width, tmp.height);
            const imageDataUrl = tmp.toDataURL('image/jpeg', 0.6);
            result = await predictEmotion(imageDataUrl);
        } else {
            const imageDataUrl = canvas.toDataURL('image/jpeg', 0.6);
            result = await predictEmotion(imageDataUrl);
        }

        // Success UI Update
        emotionLabel.textContent = result.emotion;
        console.log(result);
        confidenceLabel.textContent = `${result.confidence}%`;
        resultContainer.classList.remove('hidden');

        // Swap Buttons
        captureBtn.classList.add('hidden');
        resetBtn.classList.remove('hidden');

    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
        resultContainer.classList.remove('hidden');
        
        // Show video again if it failed
        canvas.style.display = 'none';
        video.style.display = 'block';
    } finally {
        captureBtn.classList.remove('loading');
        captureBtn.disabled = false;
    }
});

// Reset logic
resetBtn.addEventListener('click', () => {
    // Hide results
    resultContainer.classList.add('hidden');
    // Hide snapshot canvas, show live video
    canvas.style.display = 'none';
    video.style.display = 'block';
    
    // Swap buttons
    resetBtn.classList.add('hidden');
    captureBtn.classList.remove('hidden');
});

// Start the webcam when script loads
window.addEventListener('load', startWebcam);
