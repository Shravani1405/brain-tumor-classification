# Visualizing evaluation Metric Score chart
# Preload the data (already preprocessed and split into X_train, X_test, y_train, y_test)
# y_train_cat and y_test_cat are one-hot encoded
# !pip install -q keras-tuner

from keras_tuner.tuners import Hyperband

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=X_train.shape[1:]))

# Freeze base model
base_model.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(y_train_cat.shape[1], activation='softmax')(x)

model2 = Model(inputs=base_model.input, outputs=output)
model2.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Algorithm
history2 = model2.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=32)

# Predict on the model
y_pred2 = model2.predict(X_test)
y_pred2_classes = np.argmax(y_pred2, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred2_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred2_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - DL Model 2")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



def build_efficientnet_model(hp):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=X_train.shape[1:]))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)
    x = Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)
    output = Dense(y_train_cat.shape[1], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-4, 1e-5])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner2 = Hyperband(build_efficientnet_model, objective='val_accuracy', max_epochs=5, directory='tune_effnet', project_name='effnet_tuning')
tuner2.search(X_train, y_train_cat, validation_data=(X_test, y_test_cat))

best_model2 = tuner2.get_best_models(1)[0]
y_pred2_tuned = best_model2.predict(X_test)
y_pred2_tuned_classes = np.argmax(y_pred2_tuned, axis=1)
print(classification_report(y_test, y_pred2_tuned_classes))



# Save the File
# Save in HDF5 format for portability
model2.save("efficientnet_b0_brain_tumor_model.h5")

# Optionally: Save in TensorFlow SavedModel format (recommended for TF Serving / TF Lite / ONNX)
model2.save("efficientnet_saved_model", save_format="tf")

print("Model saved successfully!")

# Load the File and predict unseen data.
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
loaded_model = load_model("efficientnet_b0_brain_tumor_model.h5")
print("Model loaded successfully!")

# Example: Let's say you have one new image to test
# It must be resized, normalized just like training images
# Shape: (1, 224, 224, 3) if EfficientNetB0 expects 224x224 RGB input

# Simulating unseen image (replace this with actual image loading)
# unseen_image = actual image array of shape (224, 224, 3)
unseen_image = X_test[0]  # Using test data as unseen example (for demo)
unseen_image = np.expand_dims(unseen_image, axis=0)  # Add batch dimension

# Predict
pred_probs = loaded_model.predict(unseen_image)
pred_class = np.argmax(pred_probs)

# Assuming you have label mapping
class_labels = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}  # adjust if different

print("Predicted Class:", class_labels[pred_class])
