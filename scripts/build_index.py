import faiss
import numpy as np
import pickle
import os

def build_faiss_index(features, index_path='faiss/food_faiss.index'):
    dimension = features.shape[1]
    index = faiss.IndexFlatL2(dimension)  # exact search, L2 distance
    index.add(features)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    # Load features and meta data
    features = np.load('../faiss/features.npy').astype('float32')

    # Optionally verify dimensions
    print(f"Features shape: {features.shape}")

    # Build and save FAISS index
    build_faiss_index(features, '../faiss/food_faiss.index')
