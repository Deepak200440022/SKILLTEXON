"""
StudyBot Chatbot UI (PyQt5-based)

This application uses a trained intent classification model to create
a simple educational chatbot UI using PyQt5.

Features:
- TF-IDF vectorized input processing
- Pre-trained Keras model for intent classification
- JSON-based response mapping
- Custom UI styling using PyQt5 stylesheet
"""

import sys
import json
import pickle
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel
)
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load serialized objects
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("response_map.json", "r") as f:
    response_map = json.load(f)

# Load trained classification model
model = load_model("model.h5")

# Define English stopwords
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """
    Clean input text by:
    - Lowercasing
    - Tokenizing
    - Removing stopwords and non-alphanumeric tokens
    """
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

class ChatBotUI(QWidget):
    """
    PyQt5 GUI class for chatbot interaction.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StudyBot")
        self.setGeometry(100, 100, 600, 500)
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6fa;
                font-family: 'Segoe UI';
                font-size: 14px;
            }
            QTextEdit {
                background-color: #e9eef6;
                border: 1px solid #d0d7de;
                border-radius: 10px;
                padding: 10px;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-radius: 10px;
                padding: 8px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)

        # Layout definition
        self.layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_field = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(80)

        # Input layout (text + send)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        self.layout.addWidget(QLabel("<h2>StudyBot</h2>"))
        self.layout.addWidget(self.chat_display)
        self.layout.addLayout(input_layout)
        self.setLayout(self.layout)

        # Bind input handlers
        self.send_button.clicked.connect(self.respond)
        self.input_field.returnPressed.connect(self.respond)

    def respond(self):
        """
        Main response handler.
        Displays user input and bot response in the chat box.
        """
        user_text = self.input_field.text().strip()
        if not user_text:
            return

        # Display user message (left-aligned)
        self.chat_display.append(
            f"""
            <table width='100%'>
                <tr>
                    <td align='left'>
                        <div style="background-color:#dfe9f5;padding:6px 10px;border-radius:10px;max-width:75%;">
                            <b>You:</b> {user_text}
                        </div>
                    </td>
                    <td width='25%'></td>
                </tr>
            </table>
            """
        )

        self.input_field.clear()
        cleaned = preprocess(user_text)

        # Validate input
        if not cleaned:
            self.chat_display.append(
                f"""
                <table width='100%'>
                    <tr>
                        <td width='25%'></td>
                        <td align='right'>
                            <div style="background-color:#fcdede;padding:6px 10px;border-radius:10px;max-width:75%;">
                                <b>Bot:</b> Please enter a valid query.
                            </div>
                        </td>
                    </tr>
                </table>
                """
            )
            return

        # Predict intent
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        response = random.choice(response_map[tag])

        # Display bot message (right-aligned)
        self.chat_display.append(
            f"""
            <table width='100%'>
                <tr>
                    <td width='25%'></td>
                    <td align='right'>
                        <div style="background-color:#c6e2ff;padding:6px 10px;border-radius:10px;max-width:75%;">
                            <b>Bot:</b> {response}
                        </div>
                    </td>
                </tr>
            </table>
            """
        )

if __name__ == "__main__":
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")

    app = QApplication(sys.argv)
    window = ChatBotUI()
    window.show()
    sys.exit(app.exec_())
