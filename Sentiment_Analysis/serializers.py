# Sentiment Analysis API serializer

from rest_framework import serializers
from .models import SentimentAnalysis

class SentimentAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = SentimentAnalysis
        fields = ['id', 'text', 'prediction', 'sentiment', 'created_at']
        read_only_fields = ['prediction', 'sentiment', 'created_at']