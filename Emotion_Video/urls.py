#EMOTION_VIDEO_URLS.PY

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VideoAnalysisViewSet, analyze_video

router = DefaultRouter()
router.register(r'analyses', VideoAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('analyze/', analyze_video, name='analyze_video'),
]