import os
import tempfile
import subprocess
import numpy as np
import cv2
import hashlib
import json
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from django.conf import settings
from django.http import FileResponse
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes, action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from tensorflow.keras.models import load_model
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from .models import VideoAnalysis
from .serializers import VideoAnalysisSerializer
from Speech2Text.views import analyze_speech
from Speech2Text.models import SpeechAnalysis

# Configure folders
VIDEO_FOLDER = os.path.join(settings.MEDIA_ROOT, 'video_analysis')
TEMP_FOLDER = os.path.join(settings.MEDIA_ROOT, 'temp')
REPORTS_FOLDER = os.path.join(settings.MEDIA_ROOT, 'video_reports')
AUDIO_FOLDER = os.path.join(settings.MEDIA_ROOT, 'video_audio')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'static', 'Emotion_Image', 'emotion_model.h5')

# Ensure directories exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Ensure ffmpeg and ffprobe are in PATH
FFMPEG_PATH = os.path.join(settings.BASE_DIR, 'static', 'ffmpeg')
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Load emotion detection model
try:
    emotion_model = load_model(MODEL_PATH)
    print("Emotion model loaded successfully!")
except Exception as e:
    print(f"Error loading emotion model: {str(e)}")
    emotion_model = None

# Load Whisper model for speech recognition
try:
    import whisper
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {str(e)}")
    whisper_model = None

# Load sentiment analysis model and vectorizer
vectorizer_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'vectorizer.pkl')
sentiment_model_path = os.path.join(settings.BASE_DIR, 'static', 'Sentiment_Analysis', 'trained_model.pkl')

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

# Emotion labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if not extension:
        print(f"No extension found in filename: {filename}")
        return False
        
    if extension not in ALLOWED_EXTENSIONS:
        print(f"Extension not allowed: {extension}")
        return False
        
    return True

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

def calculate_hash(frame):
    """Calculate hash of frame to detect duplicates"""
    return hashlib.md5(frame.tobytes()).hexdigest()

def extract_audio(video_path, output_folder):
    """Extract audio (wav file) from the video file and save it"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate output audio file path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_output_path = os.path.join(output_folder, f"{video_name}_audio.wav")

    try:
        # Using FFmpeg to extract audio in WAV format with more explicit parameters
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Use PCM codec
            "-ar", "44100",  # Set sample rate
            "-ac", "2",  # Set to stereo
            audio_output_path
        ]

        # Execute the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print("Warning: FFmpeg process returned non-zero code")
            print(f"Error: {stderr.decode()}")
            
            # Try to use moviepy as a fallback
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_output_path)
                print(f"Audio extracted successfully using moviepy: {audio_output_path}")
                return audio_output_path
            except Exception as me:
                print(f"MoviePy error: {str(me)}")
                return None
        else:
            print(f"Audio extracted successfully: {audio_output_path}")
            return audio_output_path

    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None
    

def process_audio_file(audio_path):
    """Process audio file using Speech2Text functionality"""
    try:
        # Load the audio file using librosa
        print("Loading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000)
        
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
            print("No clear speech detected in the audio")
            return None, None, None, None
        
        # Generate a summary of the transcription
        summary = simple_summarize(transcription)
        print(f"Summary generated: {summary}")
        
        # Analyze sentiment of the text
        text_tfidf = loaded_vectorizer.transform([transcription])
        prediction = sentiment_model.predict(text_tfidf)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        return transcription, summary, sentiment, float(prediction)
        
    except Exception as e:
        import traceback
        print(f"Error analyzing audio: {str(e)}")
        print(traceback.format_exc())
        return None, None, None, None
        
# Replace the analyze_audio method implementation with this:
def analyze_audio(audio_path):
    """Analyze audio file: Convert speech to text and analyze sentiment"""
    # Use this if you want to reuse existing Speech2Text functionality
    try:
        # Process audio directly (similar to Speech2Text)
        return process_audio_file(audio_path)
    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        return None, None, None, None


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

def analyze_audio(audio_path):
    """Analyze audio file: Convert speech to text and analyze sentiment"""
    try:
        # Load the audio file using librosa
        print("Loading audio file...")
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        
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
            print("No clear speech detected in the audio")
            return None, None, None, None
        
        # Generate a summary of the transcription
        summary = simple_summarize(transcription)
        print(f"Summary generated: {summary}")
        
        # Analyze sentiment of the text
        text_tfidf = loaded_vectorizer.transform([transcription])
        prediction = sentiment_model.predict(text_tfidf)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        return transcription, summary, sentiment, float(prediction)
        
    except Exception as e:
        import traceback
        print(f"Error analyzing audio: {str(e)}")
        print(traceback.format_exc())
        return None, None, None, None

def extract_frames_and_analyze(video_path, temp_dir):
    """Extract frames from video and analyze emotions"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames (analyze every 5th frame to speed up processing)
    sample_rate = 5
    
    emotions = []
    timestamps = []
    hashes = set()
    frames_analyzed = 0
    
    os.makedirs(os.path.join(temp_dir, 'frames'), exist_ok=True)
    chart_path = os.path.join(temp_dir, 'emotion_chart.png')
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every nth frame
        if frame_idx % sample_rate == 0:
            # Check for duplicates
            frame_hash = calculate_hash(frame)
            if frame_hash not in hashes:
                hashes.add(frame_hash)
                
                # Predict emotion
                emotion, confidence = predict_emotion(frame)
                
                # Save important frames for visualization
                if frames_analyzed % 30 == 0:  # Save every 30th analyzed frame
                    frame_path = os.path.join(temp_dir, 'frames', f"frame_{frames_analyzed}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                # Record emotion and timestamp
                emotions.append(emotion)
                timestamps.append(frame_idx / fps)  # Convert to seconds
                frames_analyzed += 1
                
        frame_idx += 1
    
    cap.release()
    
    if len(emotions) == 0:
        raise Exception("No faces detected in the video")
        
    # Generate emotion chart
    create_emotion_chart(emotions, timestamps, chart_path)
    
    # Calculate emotion statistics
    emotion_stats = analyze_emotions(emotions, timestamps)
    
    return emotion_stats, chart_path, frames_analyzed

def analyze_emotions(emotions, timestamps):
    """Analyze emotions and calculate statistics"""
    # Count frequency of each emotion
    emotion_counts = Counter(emotions)
    total_frames = len(emotions)
    
    # Calculate percentage of each emotion
    emotion_percentages = {emotion: (count / total_frames) * 100 
                         for emotion, count in emotion_counts.items()}
    
    # Find the dominant emotion
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Unknown"
    
    # Calculate duration of each emotion
    emotion_durations = {}
    if len(emotions) > 0:
        current_emotion = emotions[0]
        start_time = timestamps[0]
        
        for i in range(1, len(emotions)):
            if emotions[i] != current_emotion or i == len(emotions) - 1:
                duration = timestamps[i] - start_time
                if current_emotion in emotion_durations:
                    emotion_durations[current_emotion] += duration
                else:
                    emotion_durations[current_emotion] = duration
                
                current_emotion = emotions[i]
                start_time = timestamps[i]
        
        # Handle the last emotion segment
        if current_emotion not in emotion_durations and len(emotions) > 0:
            emotion_durations[current_emotion] = timestamps[-1] - start_time
    
    # Calculate emotional transitions
    transitions = 0
    for i in range(1, len(emotions)):
        if emotions[i] != emotions[i-1]:
            transitions += 1
    
    # Create summary of statistics
    video_duration = timestamps[-1] - timestamps[0] if timestamps else 0
    
    emotion_stats = {
        "total_frames": total_frames,
        "video_duration_seconds": video_duration,
        "dominant_emotion": dominant_emotion,
        "emotion_counts": dict(emotion_counts),
        "emotion_percentages": emotion_percentages,
        "emotion_durations_seconds": emotion_durations,
        "emotion_transitions": transitions,
        "transition_rate_per_minute": (transitions / video_duration) * 60 if video_duration > 0 else 0
    }
    
    return emotion_stats

def create_emotion_chart(emotions, timestamps, output_path):
    """Generate a chart showing emotions over time"""
    # Convert emotions to numerical values for plotting
    emotion_to_num = {emotion: i for i, emotion in enumerate(EMOTION_LABELS)}
    emotion_nums = [emotion_to_num.get(e, -1) for e in emotions]  # -1 for unknown emotions
    
    plt.figure(figsize=(10, 6))
    
    # Plot emotions over time
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, emotion_nums, marker='.', linestyle='-', markersize=3)
    plt.yticks(range(len(EMOTION_LABELS)), EMOTION_LABELS)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Analysis Over Time')
    plt.grid(True)
    
    # Plot emotion distribution
    emotion_counts = Counter(emotions)
    plt.subplot(2, 1, 2)
    emotions_list = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    plt.bar(emotions_list, counts, color='skyblue')
    plt.ylabel('Count')
    plt.title('Emotion Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def generate_pdf_report(analysis_data, chart_path, audio_analysis, output_path):
    """Generate PDF report from analysis data"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles["Title"]
    elements.append(Paragraph("Video Analysis Report", title_style))
    elements.append(Spacer(1, 0.25 * inch))
    
    # Add timestamp
    date_style = styles["Normal"]
    date_style.alignment = 1  # Center alignment
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    elements.append(Spacer(1, 0.25 * inch))
    
    # Add summary section
    heading_style = styles["Heading2"]
    elements.append(Paragraph("Visual Analysis Summary", heading_style))
    elements.append(Spacer(1, 0.1 * inch))
    
    # Summary table data
    summary_data = [
        ["Video Duration", f"{analysis_data['video_duration_seconds']:.2f} seconds"],
        ["Frames Analyzed", str(analysis_data['total_frames'])],
        ["Dominant Emotion", analysis_data['dominant_emotion']],
        ["Emotion Transitions", str(analysis_data['emotion_transitions'])],
        ["Transition Rate", f"{analysis_data['transition_rate_per_minute']:.2f} per minute"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.25 * inch))
    
    # Add emotion distribution
    elements.append(Paragraph("Visual Emotion Distribution", heading_style))
    elements.append(Spacer(1, 0.1 * inch))
    
    # Create emotion distribution table
    emotion_data = [["Emotion", "Percentage", "Duration (seconds)"]]
    for emotion in EMOTION_LABELS:
        percentage = analysis_data['emotion_percentages'].get(emotion, 0)
        duration = analysis_data['emotion_durations_seconds'].get(emotion, 0)
        if percentage > 0:
            emotion_data.append([
                emotion,
                f"{percentage:.1f}%",
                f"{duration:.2f}"
            ])
    
    emotion_table = Table(emotion_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    emotion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(emotion_table)
    elements.append(Spacer(1, 0.25 * inch))
    
    # Add visualization
    elements.append(Paragraph("Visual Emotion Timeline", heading_style))
    elements.append(Spacer(1, 0.1 * inch))
    
    # Add chart image
    if os.path.exists(chart_path):
        img_width = 6 * inch
        img = Image(chart_path, width=img_width, height=img_width * 0.6)
        elements.append(img)
    
    # Add audio analysis if available
    if audio_analysis and audio_analysis[0]:  # Check that text is not empty
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Audio Analysis", heading_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Add audio analysis information
        transcription, summary, sentiment, sentiment_value = audio_analysis
        
        # Audio summary table
        audio_summary_data = [
            ["Speech Sentiment", sentiment],
            ["Sentiment Score", f"{sentiment_value:.2f}"]
        ]
        
        audio_summary_table = Table(audio_summary_data, colWidths=[2.5*inch, 3*inch])
        audio_summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(audio_summary_table)
        elements.append(Spacer(1, 0.15 * inch))
        
        # Add summary
        elements.append(Paragraph("Speech Summary:", styles["Heading3"]))
        elements.append(Paragraph(summary, styles["Normal"]))
        elements.append(Spacer(1, 0.15 * inch))
        
        # Add full text
        elements.append(Paragraph("Full Transcription:", styles["Heading3"]))
        elements.append(Paragraph(transcription, styles["Normal"]))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF from buffer and write to file
    buffer.seek(0)
    with open(output_path, 'wb') as f:
        f.write(buffer.read())
    
    return output_path

class VideoAnalysisViewSet(viewsets.ModelViewSet):
    queryset = VideoAnalysis.objects.all().order_by('-created_at')
    serializer_class = VideoAnalysisSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request, *args, **kwargs):
        if 'video' not in request.FILES:
            return Response({"error": "Video file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video']
        
        if not allowed_file(video_file.name):
            return Response({"error": "File format not supported"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create instance with just the video file first
        video_analysis = VideoAnalysis(video=video_file)
        video_analysis.save()
        
        # Create a temporary directory for processing
        temp_dir = os.path.join(TEMP_FOLDER, f"analysis_{video_analysis.id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get the file path
        video_path = video_analysis.video.path
        print(f"Video saved at: {video_path}")
        
        try:
            # Extract audio from video
            print("Extracting audio from video...")
            audio_path = extract_audio(video_path, AUDIO_FOLDER)
            
            # Analyze audio if extraction was successful
            audio_analysis_results = None
            
            if audio_path:
                # Save audio in database
                with open(audio_path, 'rb') as audio_file:
                    audio_filename = os.path.basename(audio_path)
                    video_analysis.audio_file.save(audio_filename, audio_file)
                
                # Analyze audio file
                print("Analyzing audio content...")
                transcription, summary, sentiment, sentiment_value = analyze_audio(audio_path)
                
                if transcription:
                    # Save audio analysis results
                    video_analysis.transcription = transcription
                    video_analysis.summary = summary
                    video_analysis.sentiment = sentiment
                    video_analysis.sentiment_value = sentiment_value
                    audio_analysis_results = (transcription, summary, sentiment, sentiment_value)
                else:
                    print("Failed to analyze audio or no speech detected")
            else:
                print("Failed to extract audio from video")
            
            # Analyze video
            print("Analyzing video emotions...")
            emotion_stats, chart_path, frames_analyzed = extract_frames_and_analyze(video_path, temp_dir)
            
            # Generate PDF report
            pdf_filename = f"video_analysis_{video_analysis.id}.pdf"
            pdf_path = os.path.join(REPORTS_FOLDER, pdf_filename)
            generate_pdf_report(emotion_stats, chart_path, audio_analysis_results, pdf_path)
            
            # Update model with analysis results
            video_analysis.dominant_emotion = emotion_stats['dominant_emotion']
            video_analysis.emotion_percentages = emotion_stats['emotion_percentages']
            video_analysis.emotion_durations = emotion_stats['emotion_durations_seconds']
            video_analysis.total_frames = emotion_stats['total_frames']
            video_analysis.video_duration = emotion_stats['video_duration_seconds']
            video_analysis.emotion_transitions = emotion_stats['emotion_transitions']
            video_analysis.transition_rate = emotion_stats['transition_rate_per_minute']
            
            # Update with PDF path
            with open(pdf_path, 'rb') as pdf_file:
                video_analysis.pdf_report.save(pdf_filename, pdf_file)
            
            video_analysis.save()
            
            # Return serialized data
            serializer = self.get_serializer(video_analysis)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            # Print detailed error information
            import traceback
            print(f"Error analyzing video: {str(e)}")
            print(traceback.format_exc())
            
            # Delete the instance if an error occurred
            video_analysis.delete()
            
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @action(detail=True, methods=['get'])
    def download_report(self, request, pk=None):
        """Download the PDF report for a video analysis"""
        video_analysis = self.get_object()
        
        if not video_analysis.pdf_report:
            return Response({"error": "PDF report not available"}, status=status.HTTP_404_NOT_FOUND)
        
        return FileResponse(
            open(video_analysis.pdf_report.path, 'rb'),
            content_type='application/pdf',
            as_attachment=True,
            filename=os.path.basename(video_analysis.pdf_report.path)
        )
        
    @action(detail=True, methods=['get'])
    def download_audio(self, request, pk=None):
        """Download the extracted audio file from the video"""
        video_analysis = self.get_object()
        
        if not video_analysis.audio_file:
            return Response({"error": "Audio file not available"}, status=status.HTTP_404_NOT_FOUND)
        
        return FileResponse(
            open(video_analysis.audio_file.path, 'rb'),
            content_type='audio/wav',
            as_attachment=True,
            filename=os.path.basename(video_analysis.audio_file.path)
        )

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def analyze_video(request):
    """
    Analyze video and generate PDF report with emotion analysis.
    Returns a JSON response with the analysis results and PDF URL.
    """
    if 'video' not in request.FILES:
        return Response({"error": "Video file is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    video_file = request.FILES['video']
    
    if not allowed_file(video_file.name):
        return Response({"error": "File format not supported"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Create instance with just the video file first
    video_analysis = VideoAnalysis(video=video_file)
    video_analysis.save()
    
    # Create a temporary directory for processing
    temp_dir = os.path.join(TEMP_FOLDER, f"analysis_{video_analysis.id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get the file path
    video_path = video_analysis.video.path
    print(f"Video saved at: {video_path}")
    
    try:
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio(video_path, AUDIO_FOLDER)
        
        # Analyze audio if extraction was successful
        audio_analysis_results = None
        speech_analysis_id = None
        
        if audio_path:
            # Save audio in database
            with open(audio_path, 'rb') as audio_file:
                audio_filename = os.path.basename(audio_path)
                video_analysis.audio_file.save(audio_filename, audio_file)
            
            # Option 1: Use our analyze_audio function
            print("Analyzing audio content...")
            transcription, summary, sentiment, sentiment_value = analyze_audio(audio_path)
            
            # Option 2: Alternatively, you can create a SpeechAnalysis record directly
            # This would leverage existing Speech2Text functionality
            # Create a temporary file-like object to send to analyze_speech
            from django.core.files.base import ContentFile
            with open(audio_path, 'rb') as f:
                audio_content = f.read()
                
            from django.core.files.uploadedfile import SimpleUploadedFile
            temp_audio = SimpleUploadedFile(
                name=os.path.basename(audio_path),
                content=audio_content,
                content_type='audio/wav'
            )
            
            # Create speech analysis instance
            speech_analysis = SpeechAnalysis(
                audio=temp_audio
            )
            
            # Process with the Speech2Text logic
            if transcription:
                speech_analysis.transcription = transcription
                speech_analysis.summary = summary
                speech_analysis.sentiment = sentiment
                speech_analysis.prediction_value = sentiment_value
                speech_analysis.save()
                speech_analysis_id = speech_analysis.id
                
                # Save audio analysis results
                video_analysis.transcription = transcription
                video_analysis.summary = summary
                video_analysis.sentiment = sentiment
                video_analysis.sentiment_value = sentiment_value
                audio_analysis_results = (transcription, summary, sentiment, sentiment_value)
            else:
                print("Failed to analyze audio or no speech detected")
        else:
            print("Failed to extract audio from video")
        
        # Analyze video
        print("Analyzing video emotions...")
        emotion_stats, chart_path, frames_analyzed = extract_frames_and_analyze(video_path, temp_dir)
        
        # Generate PDF report
        pdf_filename = f"video_analysis_{video_analysis.id}.pdf"
        pdf_path = os.path.join(REPORTS_FOLDER, pdf_filename)
        generate_pdf_report(emotion_stats, chart_path, audio_analysis_results, pdf_path)
        
        # Update model with analysis results
        video_analysis.dominant_emotion = emotion_stats['dominant_emotion']
        video_analysis.emotion_percentages = emotion_stats['emotion_percentages']
        video_analysis.emotion_durations = emotion_stats['emotion_durations_seconds']
        video_analysis.total_frames = emotion_stats['total_frames']
        video_analysis.video_duration = emotion_stats['video_duration_seconds']
        video_analysis.emotion_transitions = emotion_stats['emotion_transitions']
        video_analysis.transition_rate = emotion_stats['transition_rate_per_minute']
        
        # Update with PDF path
        with open(pdf_path, 'rb') as pdf_file:
            video_analysis.pdf_report.save(pdf_filename, pdf_file)
        
        video_analysis.save()
        
        # Return response data
        response_data = {
            "id": video_analysis.id,
            "dominant_emotion": video_analysis.dominant_emotion,
            "emotion_percentages": video_analysis.emotion_percentages,
            "video_duration": video_analysis.video_duration,
            "total_frames": video_analysis.total_frames,
            "pdf_report_url": video_analysis.pdf_report.url if video_analysis.pdf_report else None,
            "audio_file_url": video_analysis.audio_file.url if video_analysis.audio_file else None,
            
            # Audio analysis data
            "transcription": video_analysis.transcription,
            "summary": video_analysis.summary,
            "sentiment": video_analysis.sentiment,
            "sentiment_value": video_analysis.sentiment_value,
            
            # Reference to speech analysis if created
            "speech_analysis_id": speech_analysis_id
        }
        
        return Response(response_data, status=status.HTTP_201_CREATED)
    except Exception as e:
        # Print detailed error information
        import traceback
        print(f"Error analyzing video: {str(e)}")
        print(traceback.format_exc())
        
        # Delete the instance if an error occurred
        video_analysis.delete()
        
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
