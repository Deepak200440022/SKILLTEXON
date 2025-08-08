"""
Chatbot Intent Classification Training Script

This script performs the following:
1. Loads and parses intent data from a JSON file.
2. Preprocesses the text using NLTK.
3. Vectorizes input patterns using TF-IDF.
4. Encodes labels using scikit-learn's LabelEncoder.
5. Trains a simple feedforward neural network using Keras.
6. Saves the trained model, vectorizer, and label encoder.
7. Plots training metrics and logs via TensorBoard.

Dependencies:
- NLTK
- scikit-learn
- keras / tensorflow.keras
- matplotlib
- pickle
"""

import os
import json
import pickle
import keras
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Load intents JSON
with open("intents.json") as file:
    data = json.load(file)

# Extract intent tags
intents = [tag for tag in data["intents"]]

# Create response map dictionary
response_map = {tags["tag"]: tags["responses"] for tags in intents}

# Save response map to JSON
with open("response_map.json", "w") as f:
    json.dump(response_map, f)

# Create (pattern, tag) training pairs
training_data = [[pattern, entry["tag"]] for entry in intents for pattern in entry["patterns"]]
patterns, tags = zip(*training_data)

# Text preprocessing: tokenize, lowercase, remove stopwords and non-alphanumerics
patterns = [
    [word for word in word_tokenize(pattern.lower()) if word not in stopwords.words("english") and word.isalnum()]
    for pattern in patterns
]
patterns = [" ".join(pattern) for pattern in patterns]

# TF-IDF vectorization
vectorizor = TfidfVectorizer()
vectorized_patterns = vectorizor.fit_transform(patterns)

# Save TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizor, f)

# Label encoding
le = LabelEncoder()
labels = le.fit_transform(tags)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    vectorized_patterns, labels, random_state=43, shuffle=True, test_size=0.1
)

# Model definition
model = Sequential()
model.add(keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(set(labels)), activation="softmax"))

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Setup logging directory
os.makedirs("logs", exist_ok=True)

# Callbacks: Save best model, early stopping, TensorBoard logging
my_callbacks = [
    ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss', mode='min'),
    TensorBoard(log_dir='./logs'),
    EarlyStopping(patience=20, restore_best_weights=True)
]

# Train model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=my_callbacks,
    batch_size=32
)

# Plot and save accuracy curve
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.title("Accuracy over epochs")
plt.savefig("logs/accuracy.png")
plt.clf()

# Plot and save loss curve
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss over epochs")
plt.savefig("logs/loss.png")
