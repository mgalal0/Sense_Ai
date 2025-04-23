# RealTimeVideo/urls.py

from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sessions', views.RealTimeAnalysisViewSet)

urlpatterns = [
    path('start/', views.start_session, name='start_session'),
    path('process/', views.process_frame, name='process_frame'),
    path('end/', views.end_session, name='end_session'),
    path('statistics/<str:session_id>/', views.session_statistics, name='session_statistics'),
]

urlpatterns += router.urls