import streamlit as st
import cv2
import numpy as np
import joblib
import pywt
from PIL import Image

# Haar cascades load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Wavelet transform function
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Face crop function (same as training)
def get_cropped_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

# Load trained model
model = joblib.load("waleedmodel1.h5")

# Categories mapping
class_dict = {
    0: "not_User_",
    1: "User_"
}

st.title("Image Classification")
st.write("Upload an image to predict the User.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Detect and crop face
    cropped_face = get_cropped_face(img_array)

    if cropped_face is not None:
        st.image(cropped_face, caption="Detected Face", use_container_width=True)

        # Resize and wavelet
        scalled_raw_img = cv2.resize(cropped_face, (32, 32))
        img_har = w2d(cropped_face, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Combine features
        combined_img = np.vstack((
            scalled_raw_img.reshape(32*32*3, 1),
            scalled_img_har.reshape(32*32, 1)
        ))
        combined_img = combined_img.reshape(1, -1)

        # Prediction
        prediction = model.predict(combined_img)[0]
        predicted_name = class_dict[prediction]

        st.success(f"Prediction: {predicted_name}")

    else:
        st.error("No face detected or eyes not visible. Please upload a clear face image.")
