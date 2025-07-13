import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Set paths
data_dir = "flowers"  # Folder with subfolders of flower classes
img_height, img_width = 180, 180
batch_size = 32

# Load and preprocess dataset
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get number of classes
num_classes = len(train_gen.class_indices)

# Build CNN model using Functional API
inputs = keras.Input(shape=(img_height, img_width, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
cnn_model = keras.Model(inputs=inputs, outputs=outputs)

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save the CNN model
cnn_model.save("flower_cnn_model.h5")

# Feature extraction for logistic regression
feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Generate features
train_gen.reset()
features = feature_extractor.predict(train_gen)
labels = train_gen.classes

# Train logistic regression
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Save logistic regression model
joblib.dump(log_reg, "flower_logistic_regression_model.pkl")

# Print report
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))

