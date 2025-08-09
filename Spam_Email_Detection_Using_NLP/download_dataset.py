import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("marcelwiechmann/enron-spam-data")

print("Path to dataset files:", path)

target_path = "./dataset"
os.makedirs(target_path, exist_ok=True)

# Move all files
for filename in os.listdir(path):
    shutil.move(os.path.join(path, filename), os.path.join(target_path, filename))

print("Files moved to:", target_path)