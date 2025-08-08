import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QFont
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from keras.models import load_model
import cv2

class DrawingPad(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.canvas = QPixmap(self.size())
        self.canvas.fill(Qt.black)
        self.setPixmap(self.canvas)
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.white, 16, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.setPixmap(self.canvas)

    def clear(self):
        self.canvas.fill(Qt.black)
        self.setPixmap(self.canvas)

    def get_image(self):
        image = self.canvas.toImage()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        return resized

class DigitRecognizerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digit Recognizer")
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0f1c;
                color: #00d1ff;
                font-family: 'Segoe UI';
                font-size: 14px;
            }
            QPushButton {
                background-color: #0a0f1c;
                color: #00d1ff;
                border: 2px solid #00d1ff;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #1f2a3a;
            }
            QLabel#digitDisplay {
                font-size: 64px;
                color: #00d1ff;
                border: 2px solid #00d1ff;
                border-radius: 10px;
                min-width: 100px;
                min-height: 100px;
                text-align: center;
                qproperty-alignment: AlignCenter;
            }
        """)

        self.model = load_model("best_model.h5")

        self.canvas = DrawingPad()
        self.digit_display = QLabel(" ")
        self.digit_display.setObjectName("digitDisplay")

        recognize_btn = QPushButton("Recognize")
        upload_btn = QPushButton("Upload Image")

        recognize_btn.clicked.connect(self.predict)
        upload_btn.clicked.connect(self.load_image)

        hbox = QHBoxLayout()
        hbox.addWidget(recognize_btn)
        hbox.addWidget(upload_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.digit_display, alignment=Qt.AlignCenter)
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        layout.addLayout(hbox)

        self.setLayout(layout)
        self.setFixedSize(400, 500)

    def preprocess(self, img):
        img = cv2.resize(img, (28, 28))
        if np.mean(img) > 127:
            img = 255 - img
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self):
        img = self.canvas.get_image()
        processed = self.preprocess(img)
        pred = self.model.predict(processed)
        confidence = np.max(pred)
        digit = np.argmax(pred)

        if confidence < 0.75:
            self.digit_display.setText("Invalid")
        else:
            self.digit_display.setText(str(digit))

        self.canvas.clear()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                processed = self.preprocess(img)
                pred = self.model.predict(processed)
                digit = np.argmax(pred)
                self.digit_display.setText(str(digit))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizerUI()
    window.show()
    sys.exit(app.exec_())
