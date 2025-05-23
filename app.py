import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

# Load trained model
model = joblib.load("mnist_model.pkl")

# Title
st.title("MNIST Digit Recognizer")
st.write("Upload a handwritten digit image (28x28)")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Preprocess uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('L')  # convert to grayscale
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_np = np.array(img)

    # Invert colors (MNIST style)
    img_np = 255 - img_np

    # Threshold
    _, img_np = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)

    # Normalize and flatten
    img_np = img_np / 255.0
    flat_img = img_np.reshape(1, -1).astype(np.float32)

    return flat_img, img

if uploaded_file is not None:
    processed_img, display_img = preprocess_image(uploaded_file)

    # Predict
    prediction = model.predict(processed_img)[0]

    # Show results
    st.image(display_img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Digit: {prediction}")
