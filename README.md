# Brain Tumor MRI Classification with Deep Learning

This Streamlit app classifies brain MRI images into "Tumor" or "No Tumor" using a pre-trained deep learning model.

## 🚀 Features
- Upload MRI image and get prediction instantly
- Uses TensorFlow CNN model
- Clean and simple UI with Streamlit

## 📦 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure
- `app.py`: Streamlit frontend
- `utils.py`: Preprocessing and prediction helper
- `models/`: Contains the trained model file (`brain_tumor_model.h5`)

## 🧠 Model
Trained on brain MRI image dataset with binary classification (Tumor / No Tumor).