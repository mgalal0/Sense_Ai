# Speech2Text/views.py

import os
import tempfile
import whisper
import pickle
import librosa
from django.conf import settings
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import SpeechAnalysis
from .serializers import SpeechAnalysisSerializer
from datetime import datetime

# Configure audio file storage folders
RECORDINGS_FOLDER = os.path.join(settings.BASE_DIR, 'recordings')
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

# Ensure ffmpeg and ffprobe are in PATH
FFMPEG_PATH = os.path.join(settings.BASE_DIR, 'static', 'ffmpeg')
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Load Whisper model once when application starts
try:
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {str(e)}")
    whisper_model = None

# Load sentiment analysis model and vectorizer
vectorizer_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'vectorizer.pkl')
sentiment_model_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'trained_model.pkl')

# Load vectorizer and prediction model
try:
    with open(vectorizer_path, 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    with open(sentiment_model_path, 'rb') as model_file:
        sentiment_model = pickle.load(model_file)
    
    print("Sentiment analysis models loaded successfully!")
except Exception as e:
    print(f"Error loading sentiment analysis models: {str(e)}")
    loaded_vectorizer = None
    sentiment_model = None

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if not extension:
        print(f"No extension found in filename: {filename}")
        return False
        
    if extension not in ALLOWED_EXTENSIONS:
        print(f"Extension not allowed: {extension}")
        return False
        
    return True

def simple_summarize(text, ratio=0.3):
    """
    Simple text summarization without external libraries
    """
    if not text or len(text) < 100:
        return text  # Text too short to summarize
    
    # Split text into sentences manually
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ['.', '!', '?'] and current_sentence.strip():
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add the last sentence if it's not empty
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    if len(sentences) <= 3:
        return text  # Too few sentences to summarize
    
    # Calculate the number of sentences for the summary
    n_sentences = max(1, int(len(sentences) * ratio))
    
    # Take first sentence + evenly spaced sentences
    summary_sentences = [sentences[0]]  # First sentence often has important context
    
    if n_sentences > 1:
        # Choose evenly spaced sentences throughout the text
        step = len(sentences) // (n_sentences - 1)
        for i in range(1, n_sentences):
            idx = min(i * step, len(sentences) - 1)
            if sentences[idx] not in summary_sentences:  # Avoid duplicates
                summary_sentences.append(sentences[idx])
    
    # Join the summary sentences
    summary = ' '.join(summary_sentences)
    return summary

class SpeechAnalysisViewSet(viewsets.ModelViewSet):
    queryset = SpeechAnalysis.objects.all().order_by('-created_at')
    serializer_class = SpeechAnalysisSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request, *args, **kwargs):
        if 'audio' not in request.FILES:
            return Response({"error": "Audio file is required"}, status=status.HTTP_400_BAD_REQUEST)

        audio_file = request.FILES['audio']
        
        if not allowed_file(audio_file.name):
            return Response({"error": "File format not supported"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{audio_file.name}"
        file_path = os.path.join(RECORDINGS_FOLDER, unique_filename)
        
        # Save the file
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        print(f"File saved at: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        try:
            # Load the audio file using librosa
            print("Loading audio file...")
            audio, sr = librosa.load(file_path, sr=16000)
            
            print("Converting speech to text...")
            # Use Whisper with English language
            result = whisper_model.transcribe(
                audio, 
                language="en", 
                task="transcribe",
                fp16=False,
                initial_prompt="This is English text:"
            )
            transcription = result["text"].strip()
            print(f"Transcription successful: {transcription}")
            
            # Verify the text
            if len(transcription) < 2:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return Response({"error": "No clear speech detected"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate a summary of the transcription
            summary = simple_summarize(transcription)
            print(f"Summary generated: {summary}")
            
            # Analyze sentiment of the text
            text_tfidf = loaded_vectorizer.transform([transcription])
            prediction = sentiment_model.predict(text_tfidf)[0]
            sentiment = "Positive" if prediction >= 0.5 else "Negative"
            
            # Create analysis object
            speech_analysis = SpeechAnalysis(
                audio=audio_file,
                transcription=transcription,
                summary=summary,
                sentiment=sentiment,
                prediction_value=float(prediction)
            )
            speech_analysis.save()
            
            # Create serializer with saved data
            serializer = self.get_serializer(speech_analysis)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            # Print detailed error information
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            
            # Clean up file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_speech(request):
    """
    Analyze audio file: Convert speech to text then analyze sentiment
    """
    if 'audio' not in request.FILES:
        return Response({"error": "Audio file is required"}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES['audio']
    
    if not allowed_file(audio_file.name):
        return Response({"error": "File format not supported"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{audio_file.name}"
    file_path = os.path.join(RECORDINGS_FOLDER, unique_filename)
    
    # Save the file
    with open(file_path, 'wb+') as destination:
        for chunk in audio_file.chunks():
            destination.write(chunk)
    
    print(f"File saved at: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        # Load the audio file using librosa
        print("Loading audio file...")
        audio, sr = librosa.load(file_path, sr=16000)
        
        print("Converting speech to text...")
        # Use Whisper with English language
        result = whisper_model.transcribe(
            audio, 
            language="en", 
            task="transcribe",
            fp16=False,
            initial_prompt="This is English text:"
        )
        transcription = result["text"].strip()
        print(f"Transcription successful: {transcription}")
        
        # Verify the text
        if len(transcription) < 2:
            if os.path.exists(file_path):
                os.remove(file_path)
            return Response({"error": "No clear speech detected"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate a summary of the transcription
        summary = simple_summarize(transcription)
        print(f"Summary generated: {summary}")
        
        # Analyze sentiment of the text
        text_tfidf = loaded_vectorizer.transform([transcription])
        prediction = sentiment_model.predict(text_tfidf)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        # Save results in database
        speech_analysis = SpeechAnalysis(
            audio=audio_file,
            transcription=transcription,
            summary=summary,
            sentiment=sentiment,
            prediction_value=float(prediction)
        )
        speech_analysis.save()
        
        # Response with results
        response_data = {
            "id": speech_analysis.id,
            "transcription": transcription,
            "summary": summary,
            "sentiment": sentiment,
            "prediction_value": float(prediction)
        }
        
        return Response(response_data)
        
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def test_upload(request):
    """
    Test uploading an audio file without analyzing it
    To determine if the problem is with upload or processing
    """
    if 'audio' not in request.FILES:
        return Response({"error": "Audio file is required"}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES['audio']
    
    if not allowed_file(audio_file.name):
        return Response({"error": "File format not supported"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Create unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{audio_file.name}"
    file_path = os.path.join(RECORDINGS_FOLDER, unique_filename)
    
    # Save the file
    with open(file_path, 'wb+') as destination:
        for chunk in audio_file.chunks():
            destination.write(chunk)
    
    # Verify the file
    file_exists = os.path.exists(file_path)
    file_size = os.path.getsize(file_path) if file_exists else 0
    
    return Response({
        "success": True,
        "message": "File uploaded successfully",
        "file_path": file_path,
        "file_exists": file_exists,
        "file_size": file_size
    })