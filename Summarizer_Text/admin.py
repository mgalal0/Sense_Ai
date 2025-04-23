# Summarizer_Text/admin.py

from django.contrib import admin
from .models import TextSummary

@admin.register(TextSummary)
class TextSummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'summary', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('original_text', 'summary')