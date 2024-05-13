from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
import json
from SVMcheckin import predict_sample_with_svm_model

@csrf_exempt
@require_http_methods(["POST"])
def recognize_fingerprint_checkIn(request):
    if request.method == 'POST':
        try:
            # Lấy dữ liệu JSON từ request
            data_input = json.loads(request.body)

            # Đường dẫn đến mẫu vân tay và mô hình đã được huấn luyện
            sample_path = data_input['checkinImagesURL']
            model_path = data_input['modelURL']

            # Dự đoán nhãn của mẫu vân tay sử dụng mô hình đã huấn luyện
            prediction = predict_sample_with_svm_model(sample_path, model_path)

            # Trả về kết quả nhận diện vân tay dưới dạng JSON
            if prediction is not None:
                if prediction == 1:
                    return JsonResponse({'message': 'Fingerprint recognized successfully', 'result': 'true'})
                else:
                    return JsonResponse({'message': 'Fingerprint recognized successfully', 'result': 'false'})
            else:
                return JsonResponse({'error': 'Fingerprint recognition failed'}, status=500)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
