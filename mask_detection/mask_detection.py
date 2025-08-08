"""
Live Mask Detection using FaceDetectorYN and Keras Model

- Uses OpenCV's FaceDetectorYN for real-time face detection
- Classifies each detected face using a trained binary mask classifier
- Displays bounding box, class label, and confidence on webcam stream
"""

import cv2
import numpy as np
import tensorflow as tf

# Load face detection model (YuNet ONNX)
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
)

# Load trained mask classification model
model = tf.keras.models.load_model("mask_classifier_best.h5")

# Constants
IMG_SIZE = 128
LABELS = ['with_mask', 'without_mask']
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for mask, Red for no mask

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

# Stream processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))
    faces = face_detector.detect(frame)

    # If faces detected
    if faces[1] is not None:
        for face in faces[1]:
            x, y, box_w, box_h = face[:4].astype(int)
            if box_w <= 0 or box_h <= 0:
                continue

            # Extract and validate face crop
            face_crop = frame[y:y+box_h, x:x+box_w]
            if face_crop.size == 0:
                continue

            # Preprocess face for classifier
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.expand_dims(face_rgb.astype('float32') / 255.0, axis=0)

            # Predict mask status
            pred = model.predict(face_input, verbose=0)[0][0]
            class_id = int(pred >= 0.5)
            confidence = pred if class_id else 1 - pred
            label = f"{LABELS[class_id]} ({confidence:.2f})"

            # Annotate frame
            color = COLORS[class_id]
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display result
    cv2.imshow("Live Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
