import kagglehub
import os
import shutil
# Download latest version

path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
path1 = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

target_path = "./dataset"
os.makedirs(target_path, exist_ok=True)

# Move all files
for filename in os.listdir(path):
    shutil.move(os.path.join(path, filename), os.path.join(target_path, filename))
for filename in os.listdir(path1):
    shutil.move(os.path.join(path1, filename), os.path.join(target_path, filename))

print("Files moved to:", target_path)