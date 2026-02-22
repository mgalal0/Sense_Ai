<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Sense%20AI&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Multimodal%20Emotion%20%26%20Sentiment%20Intelligence%20Platform&descAlignY=55&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.36-FFD21E?style=for-the-badge)](https://huggingface.co)
[![OpenAI Whisper](https://img.shields.io/badge/Whisper-ASR-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/whisper)

<br/>

> **Sense AI** is a production-grade multimodal AI platform that understands human emotion and sentiment across images, video, audio, and text — all through a unified REST API.

<br/>

</div>

---

## What It Does

Sense AI brings together six specialized AI pipelines under one platform. Feed it an image, a video, an audio clip, or raw text — and it returns deep emotional and semantic intelligence in real time.

| Pipeline | Input | Output |
|---|---|---|
| 🖼️ Image Emotion | Face image | Emotion label + confidence |
| 🎬 Video Analysis | Video file | Full emotion timeline + audio transcription + PDF report |
| 📹 Real-Time Video | Live stream frames | Per-frame emotion with session tracking |
| 🎙️ Speech to Text | Audio file | Transcription + summary + sentiment |
| 💬 Sentiment Analysis | Text | Positive / Negative classification |
| 📝 Text Summarizer | Long text | Abstractive summary via BART |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Sense AI Platform                     │
│                     Django REST Framework                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼─────┐    ┌──────▼──────┐
    │  Image  │      │   Video   │    │  Real-Time  │
    │Emotion  │      │ Analysis  │    │   Video     │
    └────┬────┘      └─────┬─────┘    └──────┬──────┘
         │                 │                 │
    ┌────▼─────────────────▼─────────────────▼──────┐
    │         Custom CNN — emotion_model.h5          │
    │    7-class: Angry·Disgust·Fear·Happy·          │
    │            Neutral·Sad·Surprise                │
    └────────────────────────────────────────────────┘

    ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
    │  Speech2Text │    │  Sentiment   │    │  Text Summarizer   │
    │              │    │  Analysis    │    │                    │
    └──────┬───────┘    └──────┬───────┘    └─────────┬──────────┘
           │                  │                       │
    ┌──────▼──────┐    ┌──────▼───────┐    ┌──────────▼──────────┐
    │OpenAI Whisper│   │ TF-IDF +     │    │ facebook/bart-      │
    │  (base ASR) │    │sklearn Model │    │ large-cnn (BART)    │
    └─────────────┘    └──────────────┘    └─────────────────────┘
```

---

## AI Models

### 🧠 Custom Emotion CNN
A convolutional neural network trained on facial expression datasets, designed for real-time inference.

- **Architecture:** CNN with multiple Conv2D → MaxPooling → Dropout blocks, fully connected output
- **Input:** 48×48 grayscale face crops
- **Output:** 7 emotion classes with softmax probability distribution
- **Classes:** `Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`
- **Format:** Keras `.h5` — optimized for fast batch inference
- **Deployed:** Shared across Image and Video pipelines (loaded once at server startup)

### 🎙️ OpenAI Whisper
- **Model:** `whisper-base` — 74M parameters
- **Task:** Multilingual speech recognition, fine-tuned on 680K hours of audio
- **Usage:** Audio extraction from video + standalone audio files
- **Language:** English with fallback prompt engineering

### 📊 TF-IDF Sentiment Classifier
- **Pipeline:** TF-IDF vectorizer → trained scikit-learn classifier
- **Task:** Binary sentiment classification (Positive / Negative)
- **Threshold:** `prediction ≥ 0.5` → Positive
- **Shared:** Reused across Sentiment Analysis, Speech2Text, and Video Analysis pipelines

### 📝 BART Large CNN (HuggingFace)
- **Model:** `facebook/bart-large-cnn` — 400M parameters
- **Task:** Abstractive text summarization
- **Framework:** HuggingFace Transformers pipeline
- **Inference:** `do_sample=False` for deterministic output

---

## API Endpoints

```
POST   /api/emotion-image/           →  Analyze emotion from image
POST   /api/emotion-video/           →  Full video analysis + PDF report
POST   /api/realtime-video/          →  Real-time frame emotion detection
POST   /api/speech2text/             →  Transcribe + sentiment from audio
POST   /api/sentiment/               →  Text sentiment classification
POST   /api/summarizer/              →  Abstractive text summarization

GET    /api/emotion-image/{id}/      →  Retrieve past image analysis
GET    /api/emotion-video/{id}/      →  Retrieve video analysis + download PDF
GET    /api/realtime-video/{id}/     →  Retrieve session frames + results
GET    /api/speech2text/{id}/        →  Retrieve transcription result
GET    /api/sentiment/{id}/          →  Retrieve sentiment result
GET    /api/summarizer/{id}/         →  Retrieve summary result
```

### Example — Emotion from Image

```bash
curl -X POST https://your-domain/api/emotion-image/ \
  -F "image=@face.jpg"
```

```json
{
  "id": 1,
  "emotion": "Happy",
  "confidence": 0.9731,
  "created_at": "2025-02-22T14:30:00Z"
}
```

### Example — Video Full Analysis

```bash
curl -X POST https://your-domain/api/emotion-video/ \
  -F "video=@session.mp4"
```

```json
{
  "id": 5,
  "dominant_emotion": "Neutral",
  "emotion_percentages": {
    "Happy": 34.2,
    "Neutral": 41.5,
    "Sad": 12.1,
    "Angry": 8.0,
    "Fear": 4.2
  },
  "transcription": "...",
  "summary": "...",
  "sentiment": "Positive",
  "pdf_report": "/media/reports/report_5.pdf"
}
```

---

## Video Analysis Pipeline

When you send a video to Sense AI, it runs a full multimodal analysis pipeline:

```
Video File
    │
    ├──► Frame Extraction (every 5th frame, MD5 deduplicated)
    │         └──► Keras CNN → emotion per frame → timeline + statistics
    │
    ├──► Audio Extraction (ffmpeg → PCM WAV 44.1kHz)
    │         └──► Whisper ASR → transcription
    │                   └──► Extractive summarizer (30% ratio)
    │                   └──► TF-IDF + sklearn → sentiment
    │
    └──► PDF Report (ReportLab)
              ├── Emotion distribution table
              ├── Emotion-over-time chart (matplotlib)
              ├── Transcription + summary
              └── Sentiment result
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Web Framework | Django 5.2 + Django REST Framework 3.16 |
| Deep Learning | TensorFlow 2.19 · Keras 3.9 · PyTorch 2.6 |
| Face Detection | DeepFace · MTCNN · RetinaFace · MediaPipe |
| Speech Recognition | OpenAI Whisper (base) |
| NLP | HuggingFace Transformers 4.36 · NLTK 3.9 · scikit-learn 1.6 |
| Audio Processing | Librosa 0.11 · SoundFile · audioread |
| Video Processing | OpenCV 4.11 · MoviePy · ffmpeg |
| PDF Generation | ReportLab 4.3 |
| Visualization | Matplotlib |
| Server | Gunicorn + ASGI |
| Process Manager | PM2 |

---

## Getting Started

### Prerequisites
- Python 3.11+
- ffmpeg (bundled in `/static/ffmpeg/`)
- 8GB+ RAM (for loading Whisper + BART simultaneously)
- GPU recommended for video inference

### Installation

```bash
git clone https://github.com/mgalal0/Sense_Ai.git
cd Sense_Ai

python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

> Models are loaded automatically at server startup from `/static/`. No manual download needed.

---

## Team

<table>
  <tr>
    <td align="center">
      <b>Mahmoud Galal</b><br/>
      <sub>Backend & API Engineering</sub>
    </td>
    <td align="center">
      <b>Adham Ismail</b><br/>
      <sub>AI & Machine Learning Engineering</sub>
    </td>
  </tr>
</table>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
