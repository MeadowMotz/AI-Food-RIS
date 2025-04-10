### Spring 2025 Deep Learning Group Project
# FAISS Food Reverse Image Search (RIS)
Reverse image search systems allow users to find similar images based on a given input image. Such systems have widespread applications in various domains, including e-commerce, digital asset management, and social media. This project focuses on developing a reverse image search system specifically for food images. The system is designed to accept a single image input and return the top 10 visually similar images from a database of 10,000 food images.
## Checkpoint 1
Develop a reverse image search system that indexes 10,000 food images using five traditional feature extraction methods. The system should preprocess images, extract features, create a FAISS index, and enable similarity searches to retrieve the top 10 similar images based on a query image.
## Checkpoint 2
Enhance the reverse image search system by deploying it as a web service using FastAPI. This checkpoint leverages ResNet50 for feature extraction, providing a RESTful API endpoint that accepts an image and returns the top 10 similar food images.
The food image dataset can be downloaded from https://www.kaggle.com/datasets/trolukovich/food11-image-dataset.  Once downloaded, unzip archive.zip into AI-Food-RIS/data/raw/1.
# Launch instructions
Navigate to the scripts folder in your terminal and run the python scripts in the following order. 
1. download_dataset.py 
2. preprocess.py
3. extract_features.py
4. build_index.py

Or on Windows, run the batch file run.bat to do it automatically.