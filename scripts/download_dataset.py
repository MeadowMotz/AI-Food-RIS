import kagglehub
import os
import shutil

# Define the target directory
target_dir = "../data/raw"

# Check if the target directory exists, if not create it
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

if (len(os.listdir(target_dir)) == 0):
    # Download the dataset using kagglehub
    download_path = kagglehub.dataset_download("trolukovich/food11-image-dataset")

    # Move the dataset to the target directory
    shutil.move(download_path, target_dir)
else:
    print("Raw images exist.")