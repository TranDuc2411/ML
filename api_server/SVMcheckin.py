import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import joblib

def predict_sample_with_svm_model(sample_path, model_path):
    # Kiểm tra xem đường dẫn đến mẫu và mô hình có tồn tại không
    if not os.path.isfile(sample_path):
        print("Error: Sample file not found.")
        return None
    if not os.path.isfile(model_path):
        print("Error: Model file not found.")
        return None

    # Load mô hình từ file
    svm = joblib.load(model_path)

    # Đọc và tiền xử lý mẫu
    sample = imread(sample_path, as_gray=True)
    sample_resized = resize(sample, (64, 64), anti_aliasing=True)
    sample_flattened = sample_resized.flatten()
    
    # Tiêu chuẩn hóa dữ liệu mẫu
    sc = StandardScaler()
    sample_std = sc.fit_transform([sample_flattened])

    # Dự đoán nhãn của mẫu
    prediction = svm.predict(sample_std)
    
    # Trả về kết quả dự đoán
    return prediction[0]  # Trả về giá trị dự đoán đầu tiên trong mảng (chỉ có một phần tử)

# Đường dẫn đến mẫu cần dự đoán và đường dẫn đến mô hình đã được huấn luyện
sample_path = "checkin_data/101_2.tif"
model_path = "model/svm_model.pkl"

# Dự đoán nhãn của mẫu sử dụng mô hình đã huấn luyện
prediction = predict_sample_with_svm_model(sample_path, model_path)
if prediction is not None:
    if prediction == 1:
        print("The sample is classified as 'true'.")
    else:
        print("The sample is classified as 'false'.")
