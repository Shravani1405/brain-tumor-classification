
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("model/best_model.h5")

# Define class names
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# App title
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload an MRI scan and the model will predict the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {predicted_class} ({confidence*100:.2f}%)")
