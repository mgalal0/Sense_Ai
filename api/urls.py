# Api Urls.py

from django.urls import path, include
from .views import api_index , docs2

urlpatterns = [
    # API index page
    path('', api_index, name='api_index'),
    
    
    # API endpoints
    path('sentiment/', include('Sentiment_Analysis.urls')),
    path('emotion/', include('Emotion_Image.urls')),
    path('speech/', include('Speech2Text.urls')),
    path('summarize/', include('Summarizer_Text.urls')),
    path('emotion-video/', include('Emotion_Video.urls')), 
    path('realtime-video/', include('RealTimeVideo.urls')),
    path('users/', include('users.urls')),
    path('docs/', docs2.as_view(), name='docs2'),  # API documentation
    
]