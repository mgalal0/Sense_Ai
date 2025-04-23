# Emotion_Image/urls.py


from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import EmotionAnalysisViewSet, analyze_emotion

router = DefaultRouter()
router.register(r'analyses', EmotionAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('analyze/', analyze_emotion, name='analyze_emotion'),
]