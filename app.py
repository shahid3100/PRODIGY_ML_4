from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

app = Flask(__name__)

model, labels = joblib.load("gesture_model.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 64

def predict_image(img_path):
    print("🔍 Predicting image:", img_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Image read failed")
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = hog(img).reshape(1, -1)
    prediction = model.predict(features)[0]

    print("✅ Prediction index:", prediction)
    print("✅ Gesture:", labels[prediction])

    return labels[prediction]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "file" not in request.files:
            print("❌ No file part")
            return render_template("index.html", result=None)

        file = request.files["file"]

        if file.filename == "":
            print("❌ Empty filename")
            return render_template("index.html", result=None)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        print("📁 File saved:", file_path)

        filename = file.filename.lower()

        if filename.endswith((".jpg", ".jpeg", ".png")):
            result = predict_image(file_path)

        elif filename.endswith(".mp4"):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                temp_img = os.path.join(UPLOAD_FOLDER, "frame.jpg")
                cv2.imwrite(temp_img, frame)
                result = predict_image(temp_img)

        print("🎯 Final Result:", result)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
