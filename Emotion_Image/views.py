# Emotion_Image/views.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import EmotionAnalysis
from .serializers import EmotionAnalysisSerializer
import tempfile

# Load the model
model_path = os.path.join(settings.BASE_DIR, 'static', 'Emotion_Image', 'emotion_model.h5')
model = load_model(model_path)

# List of emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class EmotionAnalysisViewSet(viewsets.ModelViewSet):
    queryset = EmotionAnalysis.objects.all()
    serializer_class = EmotionAnalysisSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def perform_create(self, serializer):
        image = serializer.validated_data['image']
        
        # Save the image temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
                
        # Read and process the image
        img = cv2.imread(temp_file.name, cv2.IMREAD_GRAYSCALE)
        os.unlink(temp_file.name)  # Delete the temporary file
        
        if img is None:
            return Response({"error": "Invalid image"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Resize and preprocess the image
        img = cv2.resize(img, (48, 48))
        img = img / 255.0  # Normalize pixel values between 0 and 1
        img = img.reshape(1, 48, 48, 1)  # Reshape for the model input
        
        # Predict emotion
        prediction = model.predict(img)
        emotion_index = np.argmax(prediction)  # Get the highest confidence index
        emotion = emotion_labels[emotion_index]  # Emotion label
        confidence = float(prediction[0][emotion_index])
        
        # Save the result
        serializer.save(emotion=emotion, confidence=confidence)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_emotion(request):
    """
    Analyze emotion from an uploaded image
    """
    if 'image' not in request.FILES:
        return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

    image = request.FILES['image']
    
    # Save the image temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for chunk in image.chunks():
            temp_file.write(chunk)
            
    # Read and process the image
    img = cv2.imread(temp_file.name, cv2.IMREAD_GRAYSCALE)
    os.unlink(temp_file.name)  # Delete the temporary file
    
    if img is None:
        return Response({"error": "Invalid image"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Resize and preprocess the image
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 48, 48, 1)  # Reshape for the model
    
    # Predict emotion
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = float(prediction[0][emotion_index])
    
    # Save result to database
    emotion_analysis = EmotionAnalysis(
        image=image,
        emotion=emotion,
        confidence=confidence
    )
    emotion_analysis.save()
    
    return Response({
        "emotion": emotion,
        "confidence": confidence
    })
