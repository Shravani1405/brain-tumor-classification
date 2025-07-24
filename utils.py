import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_and_predict(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    return label, confidence