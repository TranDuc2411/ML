from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse

def recognize_fingerprint(request):
    # Logic để nhận diện vân tay ở đây
    # Trả về một phản hồi JSON với kết quả nhận diện
    return JsonResponse({'message': 'Fingerprint recognized successfully'})
