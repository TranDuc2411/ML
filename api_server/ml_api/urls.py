from django.urls import path
from . import views

urlpatterns = [
    # Định tuyến các URL của ml_api
    # Ví dụ:
    # path('get-data/', views.get_data, name='get_data'),
    path('train/', views.train_model_SVM, name='train_model_SVM'),
    path('predict/', views.predict_fingerprint, name='predict_fingerprint'),
]
