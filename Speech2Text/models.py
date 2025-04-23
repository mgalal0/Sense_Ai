# Speech2Text/models.py

from django.db import models

class SpeechAnalysis(models.Model):
    audio = models.FileField(upload_to='speech_audio/')
    transcription = models.TextField()
    summary = models.TextField(blank=True, null=True)  # Add this new field
    sentiment = models.CharField(max_length=10)
    prediction_value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio analysis - {self.id}"
    
    class Meta:
        verbose_name_plural = "Speech Analyses"