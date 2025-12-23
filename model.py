import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC

DATASET = "datasets"
IMG_SIZE = 64

X, y = [], []
labels = os.listdir(DATASET)

for idx, label in enumerate(labels):
    for img_name in os.listdir(os.path.join(DATASET, label)):
        img_path = os.path.join(DATASET, label, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        features = hog(img)
        X.append(features)
        y.append(idx)

X = np.array(X)
y = np.array(y)

model = SVC(kernel="linear")
model.fit(X, y)

joblib.dump((model, labels), "gesture_model.pkl")
print("Model trained and saved")
