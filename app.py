import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from utils import preprocess_and_predict

# Load model
model = load_model("models/brain_tumor_model.h5")

# Streamlit UI
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify it as **Tumor** or **No Tumor**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction, confidence = preprocess_and_predict(image, model)
    st.markdown(f"### 🎯 Prediction: `{prediction}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")