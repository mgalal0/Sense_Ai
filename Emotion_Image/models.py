# Emotion_Image/models.py

from django.db import models

class EmotionAnalysis(models.Model):
    image = models.ImageField(upload_to='emotion_images/')
    emotion = models.CharField(max_length=20)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image analysis - {self.emotion} ({self.confidence:.2f})"
    
    class Meta:
        verbose_name_plural = "Emotion Analyses"