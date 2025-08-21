from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),             # main page
    path("predict/", views.upload_video, name="upload_video"),  # form posts here
]
