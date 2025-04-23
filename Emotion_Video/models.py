# Emotion_Video/models.py

from django.db import models

class VideoAnalysis(models.Model):
    video = models.FileField(upload_to='video_analysis/')
    audio_file = models.FileField(upload_to='video_audio/', blank=True, null=True)
    pdf_report = models.FileField(upload_to='video_reports/', blank=True, null=True)
    
    # بيانات تحليل الفيديو
    dominant_emotion = models.CharField(max_length=20, blank=True, null=True)
    emotion_percentages = models.JSONField(default=dict, blank=True)
    emotion_durations = models.JSONField(default=dict, blank=True)
    total_frames = models.IntegerField(default=0)
    video_duration = models.FloatField(default=0)  # بالثواني
    emotion_transitions = models.IntegerField(default=0)
    transition_rate = models.FloatField(default=0)  # لكل دقيقة
    
    # بيانات تحليل الصوت
    transcription = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    sentiment = models.CharField(max_length=20, blank=True, null=True)
    sentiment_value = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video analysis - {self.id}"
    
    class Meta:
        verbose_name_plural = "Video Analyses"