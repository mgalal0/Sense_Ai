# RealTimeVideo/views.py

import os
import uuid
import base64
import numpy as np
import cv2
import json
from datetime import datetime
from collections import Counter
from tensorflow.keras.models import load_model
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .models import RealTimeAnalysis, FrameAnalysis
from .serializers import RealTimeAnalysisSerializer, FrameAnalysisSerializer

# Load the emotion detection model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'static', 'Emotion_Image', 'emotion_model.h5')
try:
    emotion_model = load_model(MODEL_PATH)
    print("Emotion model loaded successfully!")
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    emotion_model = None

# Emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_emotion(frame):
    """Predict emotion from a frame"""
    try:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to fit model input (48x48)
        resized_frame = cv2.resize(gray_frame, (48, 48))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        input_frame = normalized_frame.reshape(1, 48, 48, 1)  # Reshape for model
        
        # Predict emotion
        prediction = emotion_model.predict(input_frame)
        emotion_index = np.argmax(prediction)
        emotion = EMOTION_LABELS[emotion_index]
        
        return emotion, prediction[0][emotion_index]
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return "Unknown", 0.0

@api_view(['POST'])
@parser_classes([JSONParser])
def start_session(request):
    """Start a new real-time emotion analysis session"""
    session_id = str(uuid.uuid4())
    
    # Create new session
    session = RealTimeAnalysis(
        session_id=session_id,
        is_active=True
    )
    session.save()
    
    serializer = RealTimeAnalysisSerializer(session)
    return Response({
        "message": "Session started successfully",
        "session_data": serializer.data
    }, status=status.HTTP_201_CREATED)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def process_frame(request):
    """Process a single video frame from mobile app"""
    # Check if session_id is provided
    if 'session_id' not in request.data:
        return Response({"error": "Session ID is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Check if frame data is provided
    if 'frame_data' not in request.data and 'image' not in request.FILES:
        return Response({"error": "Frame data is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    session_id = request.data['session_id']
    frame_number = int(request.data.get('frame_number', 0))
    
    # Get session or return error
    try:
        session = RealTimeAnalysis.objects.get(session_id=session_id, is_active=True)
    except RealTimeAnalysis.DoesNotExist:
        return Response({"error": "Invalid or inactive session"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Handle different input methods (base64 or file upload)
        if 'image' in request.FILES:
            # Process uploaded image file
            image_file = request.FILES['image']
            img_array = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            # Process base64 image data
            frame_data = request.data['frame_data']
            
            # Remove header if present
            if 'data:image' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            # Decode base64 to numpy array
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return Response({"error": "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Predict emotion
        emotion, confidence = predict_emotion(frame)
        
        # Save frame analysis
        frame_analysis = FrameAnalysis(
            session=session,
            frame_number=frame_number,
            emotion=emotion,
            confidence=confidence
        )
        frame_analysis.save()
        
        # Update session statistics
        session.total_frames += 1
        
        # Update emotion percentages
        frames = FrameAnalysis.objects.filter(session=session)
        emotion_counts = Counter([f.emotion for f in frames])
        total_frames = len(frames)
        
        emotion_percentages = {emotion: (count / total_frames) * 100 
                             for emotion, count in emotion_counts.items()}
        
        # Find the dominant emotion
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Unknown"
        
        # Update session
        session.dominant_emotion = dominant_emotion
        session.emotion_percentages = emotion_percentages
        session.save()
        
        return Response({
            "frame_number": frame_number,
            "emotion": emotion,
            "confidence": float(confidence),
            "dominant_emotion": dominant_emotion,
            "emotion_percentages": emotion_percentages
        })
        
    except Exception as e:
        import traceback
        print(f"Error processing frame: {str(e)}")
        print(traceback.format_exc())
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@parser_classes([JSONParser])
def end_session(request):
    """End an active analysis session"""
    if 'session_id' not in request.data:
        return Response({"error": "Session ID is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    session_id = request.data['session_id']
    
    try:
        session = RealTimeAnalysis.objects.get(session_id=session_id, is_active=True)
        session.is_active = False
        session.end_time = datetime.now()
        session.save()
        
        serializer = RealTimeAnalysisSerializer(session)
        return Response({
            "message": "Session ended successfully",
            "session_data": serializer.data
        })
    except RealTimeAnalysis.DoesNotExist:
        return Response({"error": "Invalid or inactive session"}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def session_statistics(request, session_id):
    """Get statistics for a specific session"""
    try:
        session = RealTimeAnalysis.objects.get(session_id=session_id)
        serializer = RealTimeAnalysisSerializer(session)
        return Response(serializer.data)
    except RealTimeAnalysis.DoesNotExist:
        return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)

class RealTimeAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RealTimeAnalysis.objects.all().order_by('-start_time')
    serializer_class = RealTimeAnalysisSerializer