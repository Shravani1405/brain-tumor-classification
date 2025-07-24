import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model('model/best_model.h5')

# Class names
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Title
st.title("Brain Tumor MRI Image Classification")
st.markdown("Upload an MRI scan to classify the type of brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")