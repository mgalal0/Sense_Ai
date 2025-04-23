from django.contrib import admin
from .models import VideoAnalysis

@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'dominant_emotion', 'video_duration', 'total_frames', 'created_at')
    list_filter = ('dominant_emotion', 'created_at')
    search_fields = ('id', 'dominant_emotion')
    readonly_fields = ('emotion_percentages', 'emotion_durations', 'total_frames', 'video_duration', 
                      'emotion_transitions', 'transition_rate', 'created_at')