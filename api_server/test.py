import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score

def load_data_from_directory(directory):
    X = []
    y = []
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        if os.path.isfile(image_path) and image_name.endswith(".tif"):  # Kiểm tra xem đối tượng có phải là file hình ảnh .tif không
            image = imread(image_path, as_gray=True)
            image_resized = resize(image, (64, 64), anti_aliasing=True)
            X.append(image_resized.flatten())
            y.append(1 if 'positive_samples' in directory else 0)  # Gán nhãn 0 cho dữ liệu huấn luyện và 1 cho dữ liệu kiểm tra
        else:
            print("Warning: {} is not a valid image file, skipping.".format(image_path))
    return np.array(X), np.array(y)



# Đường dẫn đến thư mục chứa dữ liệu kiểm tra
dataset_dir = "dataset"
test_dir = os.path.join(dataset_dir, "test_data")

# Tải dữ liệu từ thư mục kiểm tra
X_test, y_test = load_data_from_directory(test_dir)

# Kiểm tra xem có dữ liệu kiểm tra không
if len(X_test) == 0 or len(y_test) == 0:
    print("Error: No testing data found.")
    exit()

# Đường dẫn đến thư mục chứa mô hình đã được huấn luyện
model_dir = "model"
model_path = os.path.join(model_dir, 'svm_model.pkl')

# Kiểm tra xem mô hình đã được huấn luyện chưa
if not os.path.isfile(model_path):
    print("Error: Model not found.")
    exit()

# Load mô hình từ file
svm = joblib.load(model_path)

# Tiêu chuẩn hóa dữ liệu kiểm tra
sc = StandardScaler()
X_test_std = sc.fit_transform(X_test)

# Sử dụng mô hình để dự đoán trên tập kiểm tra
y_pred = svm.predict(X_test_std)

# Đánh giá hiệu suất của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)
