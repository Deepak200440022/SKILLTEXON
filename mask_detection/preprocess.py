"""
Preprocess Binary Mask Dataset for Face Mask Classification

This script:
1. Loads images from two categories: 'with_mask' and 'without_mask'
2. Resizes them to a uniform size (IMG_SIZE x IMG_SIZE)
3. Normalizes pixel values to [0, 1]
4. Converts images and labels to NumPy arrays
5. Saves the processed data for later training use
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
DATASET_DIR = 'data'                # Root directory containing 'with_mask' and 'without_mask' subfolders
IMG_SIZE = 128                      # Target image size
CLASSES = ['with_mask', 'without_mask']  # Class folder names

images = []
labels = []

# Traverse each class directory
for idx, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue  # Skip if directory doesn't exist

    for filename in tqdm(os.listdir(cls_dir), desc=cls):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        path = os.path.join(cls_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue  # Skip unreadable files

        # Resize, convert color format, normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0

        images.append(img)
        labels.append(idx)  # Binary label: 0 for 'with_mask', 1 for 'without_mask'

# Convert lists to NumPy arrays
X = np.array(images, dtype='float32')  # Shape: (N, IMG_SIZE, IMG_SIZE, 3)
y = np.array(labels, dtype='float32')  # Shape: (N,)

# Summary
print(f"Dataset size: {X.shape}, Labels: {y.shape}")
print(f"Class distribution: {np.bincount(y.astype('int'))}")  # Shows [count_with_mask, count_without_mask]

# Save preprocessed data
os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/X.npy", X)
np.save("processed_data/y.npy", y)
