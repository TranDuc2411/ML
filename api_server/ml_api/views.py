from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from SVMtrain import train_svm_model
from SVMtest import evaluate_model_on_test_data

@csrf_exempt
@require_http_methods(["POST"])
def train_model_SVM(request):
    if request.method == 'POST':
        try:
            # Gọi hàm train_svm_model từ file SVMtrain.py để huấn luyện mô hình
            model_path = train_svm_model("dataset", "model")
            
            # Đánh giá hiệu suất của mô hình trên tập kiểm tra
            accuracy = evaluate_model_on_test_data("dataset", "model")
            
            if model_path:
                data = {
                    'message': 'Model trained successfully',
                    'modelURL': model_path,
                    'accuracy': accuracy
                }
                return JsonResponse(data)
            else:
                return JsonResponse({'error': 'Training failed'}, status=500)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


def predict_fingerprint(request):
    # Logic để dự đoán vân tay ở đây
    # Trả về một phản hồi JSON với kết quả dự đoán
    return JsonResponse({'message': 'Fingerprint predicted successfully'})

# from django.http import JsonResponse
from sklearn import svm
import numpy as np

def train_model1(request):
    if request.method == 'POST':
        # Lấy dữ liệu từ yêu cầu POST
        training_data = request.POST.get('TrainingData')
        test_data_true = request.POST.get('TestData_True')
        test_data_false = request.POST.get('TestData_False')

        # Chuyển đổi dữ liệu sang định dạng phù hợp để huấn luyện
        # Ở đây, chúng ta giả sử dữ liệu được đưa vào đã được chuẩn bị trước
        # Thực hiện huấn luyện SVM
        svm_model = svm.SVC()
        svm_model.fit(training_data, labels)

        # Đánh giá model bằng cách sử dụng dữ liệu kiểm tra
        accuracy_true = svm_model.score(test_data_true, true_labels)
        accuracy_false = svm_model.score(test_data_false, false_labels)

        # Trả về kết quả dưới dạng JSON
        response_data = {
            "message": "Model trained successfully",
            "accuracy_true": accuracy_true,
            "accuracy_false": accuracy_false
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
