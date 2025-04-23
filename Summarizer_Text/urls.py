# Summarizer_Text/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TextSummaryViewSet, summarize_text

router = DefaultRouter()
router.register(r'summaries', TextSummaryViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('summarize/', summarize_text, name='summarize_text'),
]