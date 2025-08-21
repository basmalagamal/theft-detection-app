import cv2
import tensorflow as tf
import numpy as np
import random

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224,224), frame_step=15, start_frame=0):
    """Extract a sequence of frames starting from start_frame."""
    result = []
    src = cv2.VideoCapture(str(video_path))
    src.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(n_frames):
        ret, frame = src.read()
        if ret:
            result.append(format_frames(frame, output_size))
            for _ in range(frame_step - 1):
                src.read()  # skip frames
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

def multiple_clips_from_video(video_path, n_frames, frame_step, output_size, clips_per_video=3):
    """Extract multiple clips from different parts of the video."""
    src = cv2.VideoCapture(str(video_path))
    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    src.release()

    clip_length = 1 + (n_frames - 1) * frame_step
    possible_starts = max(1, video_length - clip_length)

    clips = []
    for _ in range(clips_per_video):
        start_frame = random.randint(0, possible_starts)
        clip = frames_from_video_file(video_path, n_frames, output_size, frame_step, start_frame)
        clips.append(clip)

    return clips
