# theft_detection/detection/views.py

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import tempfile, os
from .model_utils import frames_from_video_file, predict

def index(request):
    return render(request, "detection/index.html")

def result(request):
    return render(request, "detection/result.html")

@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("file"):
        video_file = request.FILES["file"]

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp_path = temp.name

        try:
            # Extract frames & predict
            frames = frames_from_video_file(temp_path)
            prob = predict(frames)
            label = "Theft Detected" if prob > 0.5 else "No Theft"

            return render(request, "detection/result.html", {"label": label, "prob": prob})

        except Exception as e:
            return HttpResponse(f"❌ Error during prediction: {e}")

        finally:
            os.remove(temp_path)

    return render(request, "detection/upload.html")
