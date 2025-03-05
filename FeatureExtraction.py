import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm
import faiss
import joblib

def preprocess_images(input_dir, output_dir, size=(300, 300)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, size)
                cv2.imwrite(os.path.join(output_dir, filename), img_resized)

# Feature Extraction Functions (Already Defined Above)
# extract_sift_features, extract_hog_features, extract_lbp_features, extract_color_histogram

def extract_lbp_features(image_path, P=8, R=1, method='uniform'):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, P, R, method)
    
    # Compute the histogram
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    # Read the image
    image = cv2.imread(image_path)
    
    # Compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2],
                        None, bins,
                        [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram
    cv2.normalize(hist, hist)
    
    return hist.flatten()

def create_faiss_index(features):
    dimension = features.shape[1]
    # Choose an index type. Here, we use IndexFlatL2 for exact search.
    index = faiss.IndexFlatL2(dimension)
    index.add(features)
    return index

def build_feature_database(preprocessed_dir):
    feature_db = {}
    all_features = []
    image_ids = []
    for idx, filename in enumerate(tqdm(os.listdir(preprocessed_dir))):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(preprocessed_dir, filename)
            # Extract features
#            sift = extract_sift_features(img_path)
#            hog_feat = extract_hog_features(img_path)
            lbp_feat = extract_lbp_features(img_path)
            color_hist = extract_color_histogram(img_path)
            
            # Handle varying feature lengths
            features = []
#            if sift.size > 0:
#                features.extend(sift)
#            else:
#                features.extend([0]*128)  # Assuming SIFT descriptors are of length 128
            
#            features.extend(hog_feat)
            features.extend(lbp_feat)
            features.extend(color_hist)
            
            combined_feat = np.array(features)
            feature_db[idx] = img_path
            all_features.append(combined_feat)
            image_ids.append(idx)
    
    # Convert to numpy array
    all_features = np.array(all_features)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=300)  # Adjust n_components as needed
    reduced_features = pca.fit_transform(all_features)
    
    return reduced_features, feature_db, pca

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_meta_storage(meta_path):
    with open(meta_path, 'rb') as f:
        return pickle.load(f)

def load_pca_model(pca_path):
    return joblib.load(pca_path)

def search_similar_images(query_image_path, index, pca, meta, top_k=10):
    # Extract features from the query image
#    sift = extract_sift_features(query_image_path)
#    hog_feat = extract_hog_features(query_image_path)
    lbp_feat = extract_lbp_features(query_image_path)
    color_hist = extract_color_histogram(query_image_path)
    
    # Handle varying feature lengths
    features = []
#    if sift.size > 0:
#        features.extend(sift)
#    else:
#        features.extend([0]*128)  # Assuming SIFT descriptors are of length 128
    
#    features.extend(hog_feat)
    features.extend(lbp_feat)
    features.extend(color_hist)
    
    combined_feat = np.array(features).reshape(1, -1)
    
    # Apply PCA transformation
    reduced_feat = pca.transform(combined_feat)
    reduced_feat = np.array(reduced_feat).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(reduced_feat, top_k)
    
    # Retrieve similar image paths
    similar_images = [meta[idx] for idx in indices[0]]
    return similar_images

# The purpose of this section is copy the food images in the give directory by
# renaming them with the food type prepended to the filename.  The variable
# rootdir will contain the top level folder of the images. The copied images will
# be put under a new folder called rootdir+'_renamed'. The top level folder
# structure under rootdir will be maintained so that evaluation, training, and
# validation images will remain separate.
rename_images = True
if (rename_images):
    rootdir = 'archive'
    rootdir = Path(rootdir)
    # Return a list of regular files only, not directories
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

    for file in file_list:
        original_path = file.parts[0] + '/' + file.parts[1] + '/' + file.parts[2] + '/' + file.parts[3]
        new_path      = file.parts[0] + '_renamed/' + file.parts[1] + '/' + file.parts[2] + '_' + file.parts[3]
        try:
            shutil.copy(original_path, new_path)
        except FileNotFoundError: # raised also on missing dest parent dir
            # try creating parent directories
            print(f'Creating folder {file.parts[0] + '_renamed/' + file.parts[1]}')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy(original_path, new_path)

# The purpose of this section is to preprocess the datafiles to a uniform
# 300x300 size. Set preprocess to False if this has already been done to save
# time.
preprocess = True
if preprocess:
    output_directory = 'preprocessed_images'
    input_directory = 'archive_renamed/training'
    preprocess_images(input_directory, output_directory)

# Building the feature database and saving the indices to disk takes quite a
# bit of time.  Set reload to False if this has already been done to save time.
reload = True
if (reload):

    preprocessed_directory = 'preprocessed_images'
    features, meta_storage, pca_model = build_feature_database(preprocessed_directory)


    # Create FAISS index
    faiss_index = create_faiss_index(features)

    # Save FAISS index to disk
    faiss.write_index(faiss_index, 'food_faiss.index')

    # Save meta storage (image ID to path mapping)
    with open('meta_storage.pkl', 'wb') as f:
        pickle.dump(meta_storage, f)

    # Save PCA model for future use
    joblib.dump(pca_model, 'pca_model.pkl')


# Example usage
faiss_index = load_faiss_index('food_faiss.index')
meta_storage = load_meta_storage('meta_storage.pkl')
pca_model = load_pca_model('pca_model.pkl')

query_image = 'archive_renamed/validation/Bread_8.jpg'
top_images = search_similar_images(query_image, faiss_index, pca_model, meta_storage)

for img in top_images:
    print(img)