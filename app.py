
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model("efficientnet_b0_brain_tumor_model.h5")

model = load_trained_model()

# Define class labels
class_labels = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

# App Title
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload an MRI image and the model will predict the type of brain tumor (if any).")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Results
    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Show raw prediction scores
    st.subheader("Prediction Probabilities")
    scores_df = pd.DataFrame(prediction, columns=class_labels.values())
    st.bar_chart(scores_df.T)
