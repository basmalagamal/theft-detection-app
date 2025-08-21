from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    path('', RedirectView.as_view(url='/upload/', permanent=False)),
    path('upload/', views.upload_video, name='upload'),
    path('predict/', views.predict_video, name='predict_video'),
]
