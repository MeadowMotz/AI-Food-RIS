import faiss
import joblib
from extract_features import extract_resnet50_features
import os
import cv2

def search_similar_images(query_image_path, index, meta, top_k=10):
    features = extract_resnet50_features(query_image_path)
    features = features.reshape(1, -1)
    
    # Search in FAISS index
    distances, indices = index.search(features, top_k)
    
    # Retrieve similar image paths
    similar_images = [(meta[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return similar_images

faiss_index = faiss.read_index('../faiss/food_faiss.index')
meta_model = joblib.load('../faiss/resnet_meta.pkl')
similar_folder = '../data/similar/'
if (not os.path.exists(similar_folder)):
    os.makedirs(similar_folder)

query_image = '../data/raw/1/validation/Noodles-Pasta/0.jpg'
similar_images = search_similar_images(query_image, faiss_index, meta_model)

with open(f'{similar_folder}search_result.txt', 'w') as file:
    file.write("")
for image, score in similar_images:
    impath = os.path.abspath(image)
    with open(f'{similar_folder}search_result.txt', 'a') as file:
        file.write(f"{impath}: {score}\n")   