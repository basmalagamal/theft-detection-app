# theft_detection/detection/model_utils.py

import os
import cv2
import numpy as np
import tensorflow as tf

# ----------------- MODEL LOADING -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # detection/
MODEL_PATH = os.path.join(BASE_DIR, "model", "theft.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ----------------- PARAMETERS -----------------
n_frames = 10          # Number of frames model expects
frame_size = (224, 224)  # Image size

# ----------------- HELPER FUNCTIONS -----------------
def sample_frames(video_path, n_frames=n_frames):
    """
    Extract n evenly spaced frames from a video.
    Returns a numpy array of shape (1, n_frames, 224, 224, 3)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames.")

    frame_indices = np.linspace(0, max(total_frames - 1, 0), n_frames, dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.resize(frame, frame_size)
            frame = frame / 255.0  # Normalize
            frames.append(frame)

    cap.release()

    # Pad if video has fewer frames than n_frames
    if len(frames) < n_frames:
        pad = np.zeros((n_frames - len(frames), *frame_size, 3))
        frames = np.vstack([frames, pad])

    frames = np.expand_dims(np.array(frames), axis=0)  # Shape: (1, n_frames, 224, 224, 3)
    return frames

def predict_video(video_path, threshold=0.5):
    """
    Predict whether theft is detected in a video.
    Returns a string: "Theft Detected" or "No Theft"
    """
    if model is None:
        raise ValueError("Model is not loaded. Cannot predict.")

    frames = sample_frames(video_path)
    pred = model.predict(frames, verbose=0)[0][0]  # Sigmoid output

    return "Theft Detected" if pred > threshold else "No Theft"
