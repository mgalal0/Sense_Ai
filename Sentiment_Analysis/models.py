# Sentiment Analysis Models.py

from django.db import models

class SentimentAnalysis(models.Model):
    text = models.TextField()
    prediction = models.FloatField() # تخزين القيمة الرقمية للتنبؤ
    sentiment = models.CharField(max_length=10) # إيجابي أو سلبي
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.text[:30]}... - {self.sentiment}"
    
    class Meta:
        verbose_name_plural = "Sentiment Analyses"