# Summarizer_Text/serializers.py

from rest_framework import serializers
from .models import TextSummary

class TextSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = TextSummary
        fields = ['id', 'original_text', 'summary', 'min_length', 'max_length', 'created_at']
        read_only_fields = ['summary', 'created_at'] 