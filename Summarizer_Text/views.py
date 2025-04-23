# Summarizer_Text/views.py

from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import TextSummary
from .serializers import TextSummarySerializer
import os
import traceback
import time

# Global variable to hold the summarizer
summarizer = None

def load_summarizer():
    """Attempts to load the summarizer model"""
    global summarizer
    try:
        from transformers import pipeline
        # Print info for debugging
        print("Loading summarization model (this may take some time)...")
        start_time = time.time()
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        end_time = time.time()
        print(f"Summarization model loaded successfully in {end_time - start_time:.2f} seconds!")
        return True
    except Exception as e:
        print(f"Error loading summarization model: {str(e)}")
        print(traceback.format_exc())
        summarizer = None
        return False

# Try to load the model when the file is imported
model_loaded = load_summarizer()

def huggingface_summarize(text, min_length=None, max_length=None):
    """ Abstractive summarization using Hugging Face Transformers with optional unlimited length """
    global summarizer
    
    # Check if model is loaded
    if summarizer is None:
        # Try loading it again
        if not load_summarizer():
            raise ValueError("Summarization model could not be loaded. Please check logs for details.")
    
    # Set default values
    if min_length is None:
        min_length = 10
    if max_length is None:
        # Use a reasonable maximum length
        max_length = min(1024, len(text))
    
    try:
        summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        print(traceback.format_exc())
        raise

class TextSummaryViewSet(viewsets.ModelViewSet):
    queryset = TextSummary.objects.all().order_by('-created_at')
    serializer_class = TextSummarySerializer
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        original_text = serializer.validated_data['original_text']
        min_length = serializer.validated_data.get('min_length')
        max_length = serializer.validated_data.get('max_length')
        
        try:
            summary = huggingface_summarize(original_text, min_length, max_length)
            
            # Save to database
            text_summary = TextSummary(
                original_text=original_text,
                summary=summary,
                min_length=min_length,
                max_length=max_length
            )
            text_summary.save()
            
            # Return response
            result_serializer = self.get_serializer(text_summary)
            return Response(result_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def summarize_text(request):
    """
    API endpoint to summarize text
    """
    if 'text' not in request.data:
        return Response({"error": "Text field is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    text = request.data['text']
    min_length = request.data.get('min_length')
    max_length = request.data.get('max_length')
    
    try:
        summary = huggingface_summarize(text, min_length, max_length)
        
        # Save to database
        text_summary = TextSummary(
            original_text=text,
            summary=summary,
            min_length=min_length,
            max_length=max_length
        )
        text_summary.save()
        
        # Return response
        return Response({
            "id": text_summary.id,
            "original_text": text,
            "summary": summary,
            "min_length": min_length,
            "max_length": max_length
        })
        
    except Exception as e:
        print(f"Error summarizing text: {str(e)}")
        print(traceback.format_exc())
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)