import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model

# ------------------ PARAMETERS ------------------
HEIGHT, WIDTH = 224, 224
N_FRAMES = 10
FRAME_STEP = 15  # same as training
MODEL_PATH = os.path.join(settings.BASE_DIR, "detection/models/model.h5")

# ------------------ LOAD MODEL ------------------
# Load the flexible model (any number of frames allowed)
model = load_model(MODEL_PATH, compile=False)

# ------------------ VIDEO PREPROCESSING ------------------
def preprocess_video(video_path, n_frames=N_FRAMES, frame_step=FRAME_STEP, img_size=(HEIGHT, WIDTH)):
    """Extract frames from video and preprocess for the model."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    while len(frames) < n_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0  # normalize
            frames.append(frame)
        i += 1
    cap.release()

    # Pad with last frame if fewer than n_frames
    while len(frames) < n_frames:
        frames.append(frames[-1])

    video_array = np.array(frames, dtype="float32")
    video_array = np.expand_dims(video_array, axis=0)  # shape: (1, n_frames, H, W, 3)
    return video_array

# ------------------ VIEWS ------------------
def index(request):
    """Render the homepage with upload form."""
    return render(request, "index.html")

def predict_video(request):
    """Handle video upload, preprocess, predict, and show result."""
    context = {}
    if request.method == "POST" and request.FILES.get("file"):
        video_file = request.FILES["file"]
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        # Preprocess video
        input_data = preprocess_video(video_path)

        # Predict
        pred = model.predict(input_data)[0][0]
        label = "Shoplifter" if pred > 0.5 else "Non-shoplifter"

        context["label"] = label
        fs.delete(filename)  # optional: delete video after prediction

        return render(request, "result.html", context)

    return render(request, "index.html")
