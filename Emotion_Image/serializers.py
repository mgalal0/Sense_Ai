# Emotion_Image/Serializers.py

from rest_framework import serializers
from .models import EmotionAnalysis

class EmotionAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmotionAnalysis
        fields = ['id', 'image', 'emotion', 'confidence', 'created_at']
        read_only_fields = ['emotion', 'confidence', 'created_at']