# Summarizer_Text/models.py

from django.db import models

class TextSummary(models.Model):
    original_text = models.TextField()
    summary = models.TextField()
    min_length = models.IntegerField(null=True, blank=True)
    max_length = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Summary {self.id}: {self.summary[:50]}..."
    
    class Meta:
        verbose_name_plural = "Text Summaries"