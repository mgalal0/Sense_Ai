from django.http import HttpResponse
from django.views.generic import TemplateView

def api_index(request):
    """
    API index page with custom HTML frontend
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Graduation Project API</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f6f9;
                padding-top: 20px;
                padding-bottom: 50px;
            }
            .header {
                background: linear-gradient(135deg, #4a6bdf 0%, #57b5f9 100%);
                color: white;
                padding: 40px 0;
                margin-bottom: 40px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border-radius: 0 0 10px 10px;
            }
            .card {
                border: none;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s;
                margin-bottom: 25px;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            }
            .card-header {
                background-color: #fff;
                border-bottom: 2px solid #f0f0f0;
                font-size: 18px;
                font-weight: 600;
                color: #333;
                padding: 20px;
                border-radius: 10px 10px 0 0 !important;
            }
            .card-body {
                padding: 25px;
            }
            .endpoint-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            .endpoint-item {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
                display: flex;
                align-items: center;
                transition: all 0.2s;
            }
            .endpoint-item:last-child {
                border-bottom: none;
            }
            .endpoint-item:hover {
                background-color: #f8f9fa;
            }
            .method-badge {
                font-size: 11px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 4px;
                margin-right: 15px;
                width: 60px;
                text-align: center;
            }
            .get {
                background-color: #4caf50;
                color: white;
            }
            .post {
                background-color: #2196f3;
                color: white;
            }
            .put {
                background-color: #ff9800;
                color: white;
            }
            .delete {
                background-color: #f44336;
                color: white;
            }
            .endpoint-name {
                font-weight: 500;
                color: #444;
                font-size: 15px;
                margin-bottom: 3px;
            }
            .endpoint-url {
                font-size: 13px;
                color: #666;
            }
            footer {
                margin-top: 30px;
                text-align: center;
                color: #666;
            }
            .collection-icon {
                width: 36px;
                height: 36px;
                background-color: #e5e9ff;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 15px;
            }
            .collection-icon i {
                color: #4a69bd;
                font-size: 18px;
            }
            .card-title-wrapper {
                display: flex;
                align-items: center;
            }
            .api-description {
                max-width: 700px;
                margin: 0 auto 20px;
                text-align: center;
                color: rgba(255, 255, 255, 0.9);
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1 class="text-center">مشروع التخرج - API Documentation</h1>
                <p class="api-description">واجهة برمجة التطبيقات للتحليل العاطفي وتحويل الكلام إلى نص والميزات الأخرى</p>
            </div>
        </div>
        
        <div class="container">
            <div class="row">
                <!-- Sentiment Analysis -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title-wrapper">
                                <div class="collection-icon">
                                    <i class="fas fa-smile"></i>
                                </div>
                                <span>Sentiment Analysis</span>
                            </div>
                        </div>
                        <div class="card-body">
                            <p>Text sentiment analysis API</p>
                            <ul class="endpoint-list">
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Analyze Text</div>
                                        <div class="endpoint-url">/api/sentiment/analyze/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge get">GET</span>
                                    <div>
                                        <div class="endpoint-name">Sentiment Results</div>
                                        <div class="endpoint-url">/api/sentiment/results/</div>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Emotion Image -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title-wrapper">
                                <div class="collection-icon">
                                    <i class="fas fa-image"></i>
                                </div>
                                <span>Emotion Image</span>
                            </div>
                        </div>
                        <div class="card-body">
                            <p>Image emotion detection API</p>
                            <ul class="endpoint-list">
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Analyze Image</div>
                                        <div class="endpoint-url">/api/emotion/analyze/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge get">GET</span>
                                    <div>
                                        <div class="endpoint-name">Image Results</div>
                                        <div class="endpoint-url">/api/emotion/results/</div>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Speech to Text -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title-wrapper">
                                <div class="collection-icon">
                                    <i class="fas fa-microphone"></i>
                                </div>
                                <span>Speech to Text</span>
                            </div>
                        </div>
                        <div class="card-body">
                            <p>Audio speech to text conversion</p>
                            <ul class="endpoint-list">
                                <li class="endpoint-item">
                                    <span class="method-badge get">GET</span>
                                    <div>
                                        <div class="endpoint-name">All Analyses</div>
                                        <div class="endpoint-url">/api/speech/analyses/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Analyze Speech</div>
                                        <div class="endpoint-url">/api/speech/analyze/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Test Upload</div>
                                        <div class="endpoint-url">/api/speech/test-upload/</div>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Emotion Video -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title-wrapper">
                                <div class="collection-icon">
                                    <i class="fas fa-video"></i>
                                </div>
                                <span>Emotion Video</span>
                            </div>
                        </div>
                        <div class="card-body">
                            <p>Video emotion analysis</p>
                            <ul class="endpoint-list">
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Analyze Video</div>
                                        <div class="endpoint-url">/api/emotion-video/analyze/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge get">GET</span>
                                    <div>
                                        <div class="endpoint-name">Video Analyses</div>
                                        <div class="endpoint-url">/api/emotion-video/analyses/</div>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
                
           <!-- Real-time Video -->
<div class="col-md-6">
    <div class="card">
        <div class="card-header">
            <div class="card-title-wrapper">
                <div class="collection-icon">
                    <i class="fas fa-camera"></i>
                </div>
                <span>Real-time Video</span>
            </div>
        </div>
        <div class="card-body">
            <p>Real-time video emotion analysis API</p>
            <ul class="endpoint-list">
                <li class="endpoint-item">
                    <span class="method-badge post">POST</span>
                    <div>
                        <div class="endpoint-name">Start Session</div>
                        <div class="endpoint-url">/api/realtime-video/start/</div>
                    </div>
                </li>
                <li class="endpoint-item">
                    <span class="method-badge post">POST</span>
                    <div>
                        <div class="endpoint-name">Process Frame</div>
                        <div class="endpoint-url">/api/realtime-video/process/</div>
                    </div>
                </li>
                <li class="endpoint-item">
                    <span class="method-badge post">POST</span>
                    <div>
                        <div class="endpoint-name">End Session</div>
                        <div class="endpoint-url">/api/realtime-video/end/</div>
                    </div>
                </li>
                <li class="endpoint-item">
                    <span class="method-badge get">GET</span>
                    <div>
                        <div class="endpoint-name">Session Statistics</div>
                        <div class="endpoint-url">/api/realtime-video/statistics/{session_id}/</div>
                    </div>
                </li>
                <li class="endpoint-item">
                    <span class="method-badge get">GET</span>
                    <div>
                        <div class="endpoint-name">All Sessions</div>
                        <div class="endpoint-url">/api/realtime-video/sessions/</div>
                    </div>
                </li>
            </ul>
        </div>
    </div>
</div>
                
                <!-- Text Summarizer -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title-wrapper">
                                <div class="collection-icon">
                                    <i class="fas fa-file-alt"></i>
                                </div>
                                <span>Text Summarizer</span>
                            </div>
                        </div>
                        <div class="card-body">
                            <p>Text summarization API</p>
                            <ul class="endpoint-list">
                                <li class="endpoint-item">
                                    <span class="method-badge post">POST</span>
                                    <div>
                                        <div class="endpoint-name">Summarize Text</div>
                                        <div class="endpoint-url">/api/summarize/text/</div>
                                    </div>
                                </li>
                                <li class="endpoint-item">
                                    <span class="method-badge get">GET</span>
                                    <div>
                                        <div class="endpoint-name">Summarizer Results</div>
                                        <div class="endpoint-url">/api/summarize/results/</div>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>© Graduation Project 2025 - All Rights Reserved</p>
            </footer>
        </div>

        <script>
            // Make endpoint items clickable
            document.addEventListener('DOMContentLoaded', function() {
                const endpointItems = document.querySelectorAll('.endpoint-item');
                
                endpointItems.forEach(item => {
                    item.addEventListener('click', function() {
                        const url = this.querySelector('.endpoint-url').textContent;
                        window.location.href = url;
                    });
                    
                    // Add cursor style
                    item.style.cursor = 'pointer';
                });
            });
        </script>
    </body>
    </html>
    """
    
    return HttpResponse(html_content)

class docs2(TemplateView):
    template_name = 'real_time.html'
