# RealTimeVideo/serializers.py

from rest_framework import serializers
from .models import RealTimeAnalysis, FrameAnalysis

class FrameAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = FrameAnalysis
        fields = ['frame_number', 'timestamp', 'emotion', 'confidence']

class RealTimeAnalysisSerializer(serializers.ModelSerializer):
    frames = FrameAnalysisSerializer(many=True, read_only=True)
    
    class Meta:
        model = RealTimeAnalysis
        fields = ['id', 'session_id', 'start_time', 'end_time', 'is_active', 
                 'dominant_emotion', 'emotion_percentages', 'total_frames', 'frames']