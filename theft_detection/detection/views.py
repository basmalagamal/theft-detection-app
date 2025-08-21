# views.py
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
import tensorflow as tf
from .preprocess import frames_from_video_file

# Load your model once
MODEL_PATH = os.path.join(settings.BASE_DIR, "detection/model/theft_model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

N_FRAMES = 10
OUTPUT_SIZE = (224, 224)

def predict_theft(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        temp_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
        with open(temp_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Preprocess video
        frames = frames_from_video_file(temp_path, num_frames=N_FRAMES, output_size=OUTPUT_SIZE)
        frames = frames.astype('float32') / 255.0

        # Force shape: (1, N_FRAMES, 224,224,3)
        video_batch = np.expand_dims(frames, axis=0)

        # Predict
        pred_probs = model.predict(video_batch)
        pred_class = np.argmax(pred_probs, axis=1)[0]

        labels = {0: 'No Theft', 1: 'Theft'}
        context['prediction'] = labels.get(pred_class, 'Unknown')

        os.remove(temp_path)

    return render(request, 'detection/predict.html', context)
