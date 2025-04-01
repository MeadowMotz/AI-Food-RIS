import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
import pickle
import joblib
import os
import faiss
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, size=(300, 300)):
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

# Resize image to ensure consistency
def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)

# Extract Local Binary Patterns (LBP) features
def extract_lbp_features(image, P=8, R=1, method='uniform'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Extract Color Histogram features
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Extract SIFT features
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        return np.mean(descriptors, axis=0)
    else:
        return np.zeros(128)

# Extract Histogram of Oriented Gradients (HOG) features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feat, _ = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return hog_feat

# Collect all image paths recursively
def collect_image_paths(image_dir):
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Main execution
if __name__ == '__main__':
    output_directory = 'data/preprocessed'
    input_directory = 'data/raw/training'
    print("Preprocessing images...")
    preprocess_images(input_directory, output_directory)

    # Extract all features and combine
    feature_db = {}
    all_features = []
    image_ids = []

    print("Extracting features from images...")
    image_paths = collect_image_paths(output_directory)
    idx = 0
    for file in image_paths:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_resized = cv2.imread(file)

            sift_feat = extract_sift_features(img_resized)
            hog_feat = extract_hog_features(img_resized)
            lbp_feat = extract_lbp_features(img_resized)
            color_hist_feat = extract_color_histogram(img_resized)
            
            # Handle varying feature lengths
            features = []
            features.extend(sift_feat)
            features.extend(hog_feat)
            features.extend(lbp_feat)
            features.extend(color_hist_feat)
            
            combined_feat = np.array(features)
            feature_db[idx] = file
            all_features.append(combined_feat)
            image_ids.append(idx)
        idx = idx + 1

    # Convert to numpy array
    all_features = np.array(all_features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=300)  # Adjust n_components as needed
    reduced_features = pca.fit_transform(all_features)

    print("Creating FAISS index...")
    dimension = reduced_features.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(reduced_features)
    faiss.write_index(index, 'faiss/food_faiss.index')

    # Save meta storage (image ID to path mapping)
    with open('faiss/meta_storage.pkl', 'wb') as f:
        pickle.dump(feature_db, f)

    # Save PCA model for future use
    joblib.dump(pca, 'faiss/pca_model.pkl')

    print("Feature extraction with all methods and indexing completed successfully.")