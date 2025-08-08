import kagglehub
import shutil
import os

# Download the latest version of the chatbot dataset from KaggleHub
path = kagglehub.dataset_download("niraliivaghani/chatbot-dataset")

# Define source path (where KaggleHub downloads the dataset)
dataset_dir = os.path.join(path, "chatbot-dataset")

# Move the downloaded dataset directory to the current working directory
if os.path.exists(dataset_dir):
    shutil.move(dataset_dir, "./chatbot-dataset")

# Confirm the dataset has been moved
print("Path to dataset files:", os.path.abspath("./chatbot-dataset"))
