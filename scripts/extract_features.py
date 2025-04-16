import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import os
from tqdm import tqdm

# Determine the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Define key directories relative to project root
PREPROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', '1')
FAISS_DIR = os.path.join(PROJECT_ROOT, 'faiss')

# Specific directories/files for this script
IMAGE_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'training') # Input is preprocessed training data
OUTPUT_FEATURES_PATH = os.path.join(FAISS_DIR, 'features.npy')
OUTPUT_META_PATH = os.path.join(FAISS_DIR, 'resnet_meta.pkl') # Using .pkl based on original code


# Load ResNet50 model without the top classification layer
try:
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
except Exception as e:
    print(f"Error loading ResNet50 model: {e}")
    print("Please ensure TensorFlow and internet connection are available.")
    exit() # Exit if model cannot be loaded

def extract_resnet50_features(img_path):
    try:
        # Load the image with target size for ResNet50
        img = image.load_img(img_path, target_size=(224, 224))
    
        # Convert the image to an array
        x = image.img_to_array(img)
    
        # Expand dimensions to match the model's input shape
        x = np.expand_dims(x, axis=0)
    
        # Preprocess the image (e.g., mean subtraction)
        x = preprocess_input(x)
    
        # Extract features
        features = model.predict(x, verbose=0)
    
        # Flatten the features to create a feature vector
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features for {img_path}: {e}")
        return None # Return None on error

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
    print(f"Searching for images in: {image_dir}") # Added print
    if not os.path.isdir(image_dir): # Check if input dir exists
        print(f"Error: Input directory not found: {image_dir}")
        return image_paths
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_features(image_dir, output_meta_path='../faiss/resnet_meta.pkl') -> tuple[np.ndarray | None, dict | None]:
    """
    Extract features from preprocessed images using ResNet50.

    Args:
        image_dir (str): Path to folder containing preprocessed image files. This may be an absolute path or a path that
                         is relative to the folder this code is executed from.
        output_meta_path (ndarray): Output path at which to store the ResNet metadata pickle file.

    Returns:
    Features array and metadata dictionary, or (None, None) on error.
    """
    feature_list = []
    meta_storage = {} # Maps integer index to image path

    print("Extracting features from images...")
    image_paths = collect_image_paths(image_dir)

    if not image_paths:
        print("No images found to extract features from.")
        return None, None

    idx = 0
    for img_path in tqdm(image_paths, desc="Extracting features"):
        if img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_resnet50_features(img_path)
            if feat is not None: # Check if feature extraction was successful
                feature_list.append(feat)
                meta_storage[idx] = img_path # Store path with its index
                idx += 1 # Increment index only for successful extractions
            else:
                print(f"Skipping image due to feature extraction error: {img_path}")

    if not feature_list:
        print("No features were successfully extracted.")
        return None, None

    # Convert to numpy array
    features = np.array(feature_list).astype('float32')

    # Save meta storage (using joblib or pickle)
    try:
        # Ensure the faiss directory exists before saving
        os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
        with open(output_meta_path, 'wb') as f:
             # Using pickle as per original code name .pkl
             import pickle
             pickle.dump(meta_storage, f)
        print(f"Metadata saved to {output_meta_path}")
    except Exception as e:
        print(f"Error saving metadata to {output_meta_path}: {e}")
        return None, None # Indicate error

    return features, meta_storage

if __name__ == "__main__":
    print(f"Input image directory: {IMAGE_DIR}")
    print(f"Output features file: {OUTPUT_FEATURES_PATH}")
    print(f"Output metadata file: {OUTPUT_META_PATH}")

    # Explicitly ensure 'faiss' directory exists before saving
    os.makedirs(FAISS_DIR, exist_ok=True)

    # Run extraction if either output file is missing
    run_extraction = not os.path.exists(OUTPUT_FEATURES_PATH) or not os.path.exists(OUTPUT_META_PATH)

    if run_extraction:
        print("Extracting features...")
        features, meta = extract_features(IMAGE_DIR, output_meta_path=OUTPUT_META_PATH)

        if features is not None and meta is not None:
            # Save features array explicitly
            try:
                print(f"Saving features array (Shape: {features.shape})...")
                np.save(OUTPUT_FEATURES_PATH, features)
                print(f"Features saved to {OUTPUT_FEATURES_PATH}")
                print("Feature extraction complete.")
            except Exception as e:
                print(f"Error saving features to {OUTPUT_FEATURES_PATH}: {e}")
        else:
             print("Feature extraction failed. Check logs for errors.")
             print("Skipping feature saving.")

    else:
        print("Feature and metadata files already exist.")
        print(f"Delete '{OUTPUT_FEATURES_PATH}' and '{OUTPUT_META_PATH}' to re-run extraction.")
