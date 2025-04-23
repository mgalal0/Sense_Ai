# Sentiment Analysis API Views

from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
import pickle
import os
from django.conf import settings
from .models import SentimentAnalysis
from .serializers import SentimentAnalysisSerializer

# Load your model and vectorizer (using relative paths)
model_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'trained_model.pkl')
vectorizer_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'vectorizer.pkl')

# Load the vectorizer and prediction model
with open(vectorizer_path, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

class SentimentAnalysisViewSet(viewsets.ModelViewSet):
    queryset = SentimentAnalysis.objects.all()
    serializer_class = SentimentAnalysisSerializer
    
    def perform_create(self, serializer):
        # Extract the text from the submitted data
        text = serializer.validated_data['text']
        
        # Transform the text using the loaded vectorizer
        text_tfidf = loaded_vectorizer.transform([text])
        
        # Predict using the loaded model
        prediction = loaded_model.predict(text_tfidf)[0]
        
        # Determine sentiment based on prediction value
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        # Save the instance with prediction results
        serializer.save(prediction=prediction, sentiment=sentiment)

@api_view(['POST'])
def analyze_sentiment(request):
    """
    API endpoint to analyze sentiment of a given text.
    Accepts a POST request with a 'text' field in the body.
    """
    if 'text' not in request.data:
        return Response({"error": "The 'text' field is required."}, status=status.HTTP_400_BAD_REQUEST)

    text = request.data['text']
    
    # Transform the text using the vectorizer
    text_tfidf = loaded_vectorizer.transform([text])
    
    # Predict using the model
    prediction = loaded_model.predict(text_tfidf)[0]
    
    # Determine sentiment
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    
    # Create a record in the database (optional)
    sentiment_analysis = SentimentAnalysis(
        text=text,
        prediction=float(prediction),
        sentiment=sentiment
    )
    sentiment_analysis.save()
    
    return Response({
        "text": text,
        "prediction": float(prediction),
        "sentiment": sentiment
    })
