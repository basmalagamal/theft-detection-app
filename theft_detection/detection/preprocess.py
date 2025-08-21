# detection/preprocess.py
import cv2
import numpy as np

def frames_from_video_file(video_path, n_frames=10, output_size=(224, 224), frame_step=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            frame = cv2.resize(frame, output_size)
            frames.append(frame)
        count += 1
    cap.release()
    while len(frames) < n_frames:
        frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
    return np.array(frames)
