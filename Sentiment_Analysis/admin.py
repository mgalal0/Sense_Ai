from django.contrib import admin
from .models import SentimentAnalysis

@admin.register(SentimentAnalysis)
class SentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ('text', 'sentiment', 'prediction', 'created_at')
    list_filter = ('sentiment', 'created_at')
    search_fields = ('text',)