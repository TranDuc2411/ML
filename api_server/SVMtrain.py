import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def train_svm_model(dataset_dir, model_dir):
    def load_data_from_directory(directory):
        X = []
        y = []
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith(".tif"):  # Kiểm tra tệp có đuôi là .tif không
                        image_path = os.path.join(class_dir, image_name)
                        if os.path.isfile(image_path):  
                            image = imread(image_path, as_gray=True)
                            image_resized = resize(image, (64, 64), anti_aliasing=True)
                            X.append(image_resized.flatten())
                            y.append(1 if class_name == 'positive_samples' else 0)  # Gán nhãn 1 cho dữ liệu positive_samples và 0 cho negative_samples
                        else:
                            print("Warning: {} is not a file, skipping.".format(image_path))
            else:
                print("Warning: {} is not a directory, skipping.".format(class_dir))
        return np.array(X), np.array(y)

    # Tải dữ liệu từ các thư mục
    train_dir = os.path.join(dataset_dir, "train_data")
    X_train, y_train = load_data_from_directory(train_dir)

    # Kiểm tra xem có dữ liệu huấn luyện không
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: No training data found.")
        return None

    # Tiêu chuẩn hóa dữ liệu
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    # Kiểm tra xem có ít nhất hai lớp trong dữ liệu huấn luyện không
    if len(np.unique(y_train)) < 2:
        print("Error: Training data must contain samples from at least two classes.")
        return None

    # Khởi tạo mô hình SVM và huấn luyện
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train_std, y_train)

    # Lưu mô hình vào thư mục model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    joblib.dump(svm, model_path)
    print("Model saved to:", model_path)
    return model_path

# Đường dẫn đến thư mục chứa dữ liệu và thư mục để lưu mô hình
dataset_dir = "dataset"
model_dir = "model"

# Huấn luyện mô hình SVM và lưu mô hình
trained_model_path = train_svm_model(dataset_dir, model_dir)
if trained_model_path is None:
    print("Training failed.")
