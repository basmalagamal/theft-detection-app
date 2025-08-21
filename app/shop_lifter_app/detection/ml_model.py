# detection/ml_model.py
import tensorflow as tf
import os
# Load trained model (change filename if .h5 or .keras)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_model(input_shape=(10, 224, 224, 3), num_classes=1):
    base_cnn = MobileNetV2(include_top=False, weights="imagenet",
                           pooling='avg', input_shape=(224, 224, 3))
    base_cnn.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.TimeDistributed(base_cnn)(inputs)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "theft_model.h5")   # model is in project root

model = build_model(input_shape=(None, 224, 224, 3))   # <-- allow flexible timesteps
model.load_weights(MODEL_PATH)
model.save("theft_model.keras")  # Save in new format
model = tf.keras.models.load_model("theft_model.keras")

# def predict(video_frames):
#     """
#     video_frames: numpy array shaped (n_frames, 224, 224, 3)
#     Returns probability of shoplifting
#     """
#     video_frames = video_frames / 255.0  # normalize
#     video_frames = video_frames.reshape(1, *video_frames.shape)  # add batch dim
#     prediction = model.predict(video_frames)[0][0]
#     return prediction


def predict(video_frames):
    video_frames = video_frames.reshape(1, *video_frames.shape)
    prediction = model.predict(video_frames)[0][0]
    return prediction
