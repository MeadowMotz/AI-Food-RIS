import cv2
import os

def preprocess_images(input_dir, output_dir, size=(300, 300)):
    """
    Preprocesses all image files in input_dir, including subfolders, and puts the processed images in output_dir.

    Args:
        input_dir (str): Path to the image files to be preprocessed. This may be an absolute path or a path that is
                         relative to the folder this code is executed from.
        output_dir (str): Path to the output folder where the preprocessed images are to be stored. All preprocessed
                          images will be stored in the same folder structure as the input directory. This may be an
                          absolute path or a path that is relative to the folder this code is executed from.
    """
    image_paths = collect_image_paths(input_dir)
    relPathLoc = len(input_dir)
    for file in image_paths:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            relPath = file[relPathLoc:]
            new_path = output_dir + relPath
            img = cv2.imread(file)
            if img is not None:
                img_resized = cv2.resize(img, size)
                result = cv2.imwrite(new_path, img_resized)
                if (result == False):
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    result = cv2.imwrite(new_path, img_resized)

# Collect all image paths recursively
def collect_image_paths(image_dir):
    """
    Collect all image paths recursively.

    Args:
        image_dir (str): Path to folder containing image files. This may be an absolute path or a path that is relative
                         to the folder this code is executed from.

    Returns:
    list: List containing the paths to all image files in image_dir and all of its subdirectories. 
    """
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    # Directories are relative to the scripts folder where this script should be executed.
    output_directory = '../data/preprocessed'
    input_directory = '../data/raw/training'
    preprocess_images(input_directory, output_directory)
