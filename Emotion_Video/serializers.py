# Emotion_Video/serializers.py

from rest_framework import serializers
from .models import VideoAnalysis

class VideoAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoAnalysis
        fields = [
            'id', 'video', 'audio_file', 'pdf_report', 
            'dominant_emotion', 'emotion_percentages', 'emotion_durations', 
            'total_frames', 'video_duration', 'emotion_transitions', 'transition_rate',
            'transcription', 'summary', 'sentiment', 'sentiment_value',
            'created_at'
        ]
        read_only_fields = [
            'audio_file', 'pdf_report', 
            'dominant_emotion', 'emotion_percentages', 'emotion_durations', 
            'total_frames', 'video_duration', 'emotion_transitions', 'transition_rate',
            'transcription', 'summary', 'sentiment', 'sentiment_value',
            'created_at'
        ]