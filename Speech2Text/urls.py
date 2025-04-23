# Speech2Text/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SpeechAnalysisViewSet, analyze_speech, test_upload

router = DefaultRouter()
router.register(r'analyses', SpeechAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('analyze/', analyze_speech, name='analyze_speech'),
    path('test-upload/', test_upload, name='test_upload'),
]