import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
import pickle
import joblib
import os
import faiss
from tqdm import tqdm

# Load ResNet50 model without the top classification layer
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet50_features(img_path):
    # Load the image with target size for ResNet50
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert the image to an array
    x = image.img_to_array(img)
    
    # Expand dimensions to match the model's input shape
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image (e.g., mean subtraction)
    x = preprocess_input(x)
    
    # Extract features
    features = model.predict(x)
    
    # Flatten the features to create a feature vector
    return features.flatten()

def extract_features(image_dir, output_meta='../faiss/resnet_meta.pkl') -> np.ndarray:
    """
    Extract features from preprocessed images using ResNet50.

    Args:
        image_dir (str): Path to folder containing preprocessed image files. This may be an absolute path or a path that
                         is relative to the folder this code is executed from.
        output_meta (ndarray): Output path at which to store the ResNet metadata pickle file.

    Returns:
    numpy array: List containing the paths to all image files in image_dir and all of its subdirectories. 
    """
    # Ensure the faiss directory exists before saving
    os.makedirs(os.path.dirname(output_meta), exist_ok=True)

    feature_list = []
    meta_storage = {}

    print("Extracting features from images...")
    image_paths = collect_image_paths(image_dir)
    idx = 0
    for img_path in image_paths:
        if img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_resnet50_features(img_path)
            feature_list.append(feat)
            meta_storage[idx] = img_path
            idx = idx + 1

    # Convert to numpy array
    features = np.array(feature_list).astype('float32')
    
    # Save meta storage
    with open(output_meta, 'wb') as f:
        pickle.dump(meta_storage, f)

    return features

# Collect all image paths recursively
def collect_image_paths(image_dir) -> list:
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
    image_directory = '../data/1/preprocessed/training'
    output_features = '../faiss/features.npy'
    output_meta = '../faiss/resnet_meta.pkl'

    # Explicitly ensure 'faiss' directory exists before saving
    os.makedirs(os.path.dirname(output_features), exist_ok=True)

    if (not os.path.exists(output_features) or not os.path.exists(output_meta)):
        print("Extracting features...")
        features = extract_features(image_directory, output_meta=output_meta)

        # Save features array explicitly
        print("Saving features array...")
        np.save(output_features, features)
    else:
        print("Extracted features exist.")
