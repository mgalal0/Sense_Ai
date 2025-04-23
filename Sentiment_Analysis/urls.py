# Sentiment Analysis Django App URL Configuration


from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SentimentAnalysisViewSet, analyze_sentiment

router = DefaultRouter()
router.register(r'analysis', SentimentAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('analyze/', analyze_sentiment, name='analyze_sentiment'),
]