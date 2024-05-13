from django.urls import path
from . import views

urlpatterns = [
    # Định tuyến các URL của fingerprint_api
    # Ví dụ:
    path('checkin/', views.recognize_fingerprint_checkIn, name='recognize_fingerprint_checkin'),
]
