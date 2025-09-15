import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_isl_dataset(data_dir, img_size=(64, 64), test_size=0.2):
    images = []
    labels = []

    classes = sorted(os.listdir(data_dir))

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels)

    return X_train, X_test, y_train, y_test, lb
