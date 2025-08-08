"""
Train a Binary Mask Detection CNN using Keras

This script:
1. Loads the preprocessed dataset from NumPy arrays
2. Splits it into training and validation sets
3. Defines a CNN model for binary classification (with_mask / without_mask)
4. Trains the model with appropriate callbacks
5. Saves training plots (accuracy and loss)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load full dataset
X = np.load('processed_data/X.npy')  # Image data
y = np.load('processed_data/y.npy')  # Binary labels

print("Full dataset:", X.shape, y.shape)

# Split into train and validation sets (80%/20%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

# Ensure TensorFlow backend for Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Build CNN model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 2
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 3
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # Output for binary classification

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callback to save best model
checkpoint_best = ModelCheckpoint(
    filepath="mask_classifier_best.h5",
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

# Training callbacks
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5),
    checkpoint_best
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Plot training metrics
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Save training metrics to file
plt.tight_layout()
plt.savefig("training_metrics.png")
print("Saved training graph → training_metrics.png")
