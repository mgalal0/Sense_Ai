# RealTimeVideo/models.py
import os
import uuid
from django.db import models

def get_file_path(instance, filename):
    """Generate a unique file path for uploaded files"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('realtime_videos', filename)

class RealTimeAnalysis(models.Model):
    session_id = models.CharField(max_length=64, unique=True)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    video_file = models.FileField(upload_to=get_file_path, null=True, blank=True)
    
    # Aggregate statistics
    dominant_emotion = models.CharField(max_length=20, null=True, blank=True)
    emotion_percentages = models.JSONField(default=dict, blank=True)
    total_frames = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Session {self.session_id} - {self.start_time}"
    
    class Meta:
        verbose_name_plural = "Real-time Analyses"

class FrameAnalysis(models.Model):
    session = models.ForeignKey(RealTimeAnalysis, on_delete=models.CASCADE, related_name='frames')
    timestamp = models.DateTimeField(auto_now_add=True)
    frame_number = models.IntegerField(default=0)
    emotion = models.CharField(max_length=20, default="Unknown")
    confidence = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"Frame {self.frame_number} - {self.emotion}"