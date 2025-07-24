# brain-tumor-classification
A deep learning–powered Streamlit web app that classifies MRI brain scans into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor using an optimized EfficientNetB0 model. Built for fast, accurate, and accessible tumor detection to assist medical professionals and patients. Easily deployable locally or on Streamlit Cloud.

#  Brain Tumor MRI Classification using Deep Learning


A deep learning–based Streamlit web app that classifies brain tumors from MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.



---

##  Problem Statement

Brain tumors are one of the deadliest neurological conditions. Early detection using MRI can help improve treatment planning and patient survival rates. This project aims to automate tumor detection and classification using advanced deep learning models and provide doctors and patients with a simple, accessible web app interface.

---

##  Project Highlights

-  Built with **EfficientNetB0** CNN for high accuracy and efficiency  
-  Preprocessed MRI images using Keras utilities  
-  Streamlit-based interactive UI  
-  Evaluation metrics: Accuracy, Precision, Recall, F1-Score  
-  Ready for deployment on **Streamlit Cloud**  

---

##  Tech Stack

| Tool | Description |
|------|-------------|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Deep Learning model implementation |
| **EfficientNetB0** | Pretrained model fine-tuned for classification |
| **Pillow, Matplotlib** | Image processing & visualization |
| **Streamlit** | Interactive web application framework |

---

##  Project Structure
---

##  Model Input & Output

- **Input:** MRI image (`.jpg`, `.jpeg`, `.png`)
- **Output:** Predicted tumor type and confidence score
- **Classes:** 
  - `Glioma`
  - `Meningioma`
  - `Pituitary`
  - `No Tumor`

---

##  Model Performance

| Metric     | Score  |
|------------|--------|
| Accuracy   | 97.6%  |
| Precision  | 97.8%  |
| Recall     | 97.4%  |
| F1-Score   | 97.6%  |

*Based on validation set from the Brain Tumor Dataset.*

---

##  Installation & Local Deployment

=======
## Features
- Upload MRI image and get prediction instantly
- Uses TensorFlow CNN model
- Clean and simple UI with Streamlit

##  How to Run
>>>>>>> 7d46ae7ed01664d89f9606118448178329595faf
```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

<<<<<<< HEAD
=======
##  Structure
- `app.py`: Streamlit frontend
- `utils.py`: Preprocessing and prediction helper
- `models/`: Contains the trained model file (`brain_tumor_model.h5`)

##  Model
Trained on brain MRI image dataset with binary classification (Tumor / No Tumor).
>>>>>>> 7d46ae7ed01664d89f9606118448178329595faf
