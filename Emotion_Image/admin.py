# Emotion_Image/admin.py

from django.contrib import admin
from .models import EmotionAnalysis

@admin.register(EmotionAnalysis)
class EmotionAnalysisAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'confidence', 'created_at')
    list_filter = ('emotion', 'created_at')