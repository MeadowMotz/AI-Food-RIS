import kagglehub
import os
import shutil

# Define the target directory
target_dir = "../data/raw"

# Remove the output directory if it exists to clear out existing files
if os.path.exists(target_dir):
    if len(os.listdir(target_dir)) != 0:
        shutil.rmtree(target_dir)

# Check if the target directory exists, if not create it
# Use exist_ok=True to avoid errors if it already exists
os.makedirs(target_dir, exist_ok=True)

# Download the dataset using kagglehub
download_path = kagglehub.dataset_download("trolukovich/food11-image-dataset")

# Move the dataset to the target directory
shutil.move(download_path, target_dir)
