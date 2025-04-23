# Spech2Text/admin.py

from django.contrib import admin
from .models import SpeechAnalysis

@admin.register(SpeechAnalysis)
class SpeechAnalysisAdmin(admin.ModelAdmin):
    list_display = ('transcription', 'sentiment', 'prediction_value', 'created_at')
    list_filter = ('sentiment', 'created_at')
    search_fields = ('transcription',)