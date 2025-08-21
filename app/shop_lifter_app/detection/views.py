from django.shortcuts import render

# Create your views here.
# detection/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2, numpy as np, tempfile
from .ml_model import predict
from .utils import frames_from_video_file  # reuse your frame extractor

@csrf_exempt
def predict_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        # Save uploaded video temporarily
        video_file = request.FILES["video"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp_path = temp.name

        # Extract frames
        frames = frames_from_video_file(temp_path, n_frames=10, frame_step=15)

        # Predict
        prob = predict(frames)

        result = {
            "probability": float(prob),
            "label": "Shoplifter" if prob > 0.5 else "Non-Shoplifter"
        }
        return JsonResponse(result)

    return JsonResponse({"error": "Upload a video file"}, status=400)

from django.http import HttpResponse

from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os

def home(request):
    return HttpResponse("🚀 Shop Lifter App is running! Upload your video at /upload/")

def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video = request.FILES["video"]
        fs = FileSystemStorage(location="media/uploads")
        filename = fs.save(video.name, video)
        file_path = fs.path(filename)  # full path to file
        video_url = fs.url(filename)

        # === Extract frames and predict ===
        frames = frames_from_video_file(file_path, n_frames=10, frame_step=15)
        prob = predict(frames)
        label = "Shoplifter" if prob > 0.5 else "Non-Shoplifter"

        return render(request, "detection/result.html", {
            "video_url": video_url,
            "label": label,
            "prob": round(prob, 2),
        })

    return render(request, "detection/upload.html")


