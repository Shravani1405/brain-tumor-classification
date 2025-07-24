# DL Model - 1 Implementation (Basic CNN)
# !pip install -q keras-tuner

from keras_tuner.tuners import RandomSearch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm

# Define the image size
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Load preprocessed data (replace with actual image loading)
dataset_path = '/content/drive/MyDrive/TumourDataset/Tumour/train/datasets' # Update path to the actual image data
categories = os.listdir(dataset_path)

images = []
labels = []

# Use tqdm for progress bar
for category in tqdm(categories, desc="Loading images"):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB') # Ensure images are in RGB format
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")


X = np.array(images)
# Convert labels to numerical format
label_dict = {category: i for i, category in enumerate(categories)}
y = np.array([label_dict[label] for label in labels])


# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Build CNN Model
model1 = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)), # Define the input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train_cat.shape[1], activation='softmax')
])


model1.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Algorithm
history1 = model1.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=32)


# Predict on the model
y_pred1 = model1.predict(X_test)
y_pred1_classes = np.argmax(y_pred1, axis=1)

# Evaluate the model
print("\nModel 1 Evaluation:")
print(classification_report(y_test, y_pred1_classes, target_names=categories))

# Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix - Model 1")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy Score
accuracy1 = accuracy_score(y_test, y_pred1_classes)
print(f"Accuracy Score - Model 1: {accuracy1}")


# DL Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm

# Predict on the model
# NOTE: Keras models don't work with sklearn GridSearchCV by default.
# We'll use Keras Tuner for tuning.


def build_model(hp):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values=[3,5]),
        activation='relu',
        input_shape=X_train.shape[1:]
    ))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=256, step=32),
        activation='relu'
    ))
    model.add(Dense(y_train_cat.shape[1], activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='model_tuning',
    project_name='dl_model_1'
)

tuner.search(X_train, y_train_cat, epochs=5, validation_data=(X_test, y_test_cat))

# Best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Predict using tuned model
y_pred1_tuned = best_model.predict(X_test)
y_pred1_tuned_classes = np.argmax(y_pred1_tuned, axis=1)

# Evaluate
print(classification_report(y_test, y_pred1_tuned_classes))