# Speech2Text/serializers.py

from rest_framework import serializers
from .models import SpeechAnalysis

class SpeechAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpeechAnalysis
        fields = ['id', 'audio', 'transcription', 'summary', 'sentiment', 'prediction_value', 'created_at']
        read_only_fields = ['transcription', 'summary', 'sentiment', 'prediction_value', 'created_at']