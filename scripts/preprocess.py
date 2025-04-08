import cv2
import os
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, size=(300, 300)):
    image_paths = collect_image_paths(input_dir)
    print(f"Found {len(image_paths)} images for processing.")
    
    for file in tqdm(image_paths):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            # explicitly ensure correct relative path
            rel_path = os.path.relpath(file, input_dir).lstrip(os.sep)
            new_path = os.path.join(output_dir, rel_path)

            # ensure directory exists BEFORE writing
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            img = cv2.imread(file)
            if img is not None:
                img_resized = cv2.resize(img, size)
                success = cv2.imwrite(new_path, img_resized)
                if not success:
                    print(f"Failed to save image: {new_path}")

def collect_image_paths(image_dir):
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    # Explicitly use absolute paths relative to script's own location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    input_directory = os.path.join(base_dir, 'data', 'raw', '1', 'training')
    output_directory = os.path.join(base_dir, 'data', 'preprocessed', '1', 'training')

    if (not os.path.exists(output_directory)):
        preprocess_images(input_directory, output_directory)
    else:
        print("Preprocessed images already exist.")