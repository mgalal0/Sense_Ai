# RealTimeVideo/admin.py

from django.contrib import admin
from .models import RealTimeAnalysis, FrameAnalysis

@admin.register(RealTimeAnalysis)
class RealTimeAnalysisAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'start_time', 'end_time', 'is_active', 'dominant_emotion', 'total_frames')
    list_filter = ('is_active', 'dominant_emotion')
    search_fields = ('session_id',)

@admin.register(FrameAnalysis)
class FrameAnalysisAdmin(admin.ModelAdmin):
    list_display = ('session', 'frame_number', 'timestamp', 'emotion', 'confidence')
    list_filter = ('emotion',)
    search_fields = ('session__session_id',)