import os
import subprocess
import logging
import traceback
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# --- Add necessary imports ---
import numpy as np
import faiss
import joblib # Or import pickle if you saved meta with pickle
import pickle # Using pickle based on previous refactoring
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
# --- End Add necessary imports ---


# Set up logging
# Use logging.DEBUG to see the debug messages we added
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Set a secret key for Flask-SocketIO (replace 'your secret key' with a real secret)
app.config['SECRET_KEY'] = 'your secret key'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Define Project Paths Relative to server.py ---
# app.root_path is the path to the directory where server.py is (e.g., .../AI-Food-RIS-main/app)
PROJECT_ROOT = os.path.abspath(os.path.join(app.root_path, os.pardir))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
STATIC_FOLDER = app.static_folder # Use Flask's default static folder if needed
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FAISS_DIR = os.path.join(PROJECT_ROOT, "faiss")
INDEX_PATH = os.path.join(FAISS_DIR, 'food_faiss.index')
META_PATH = os.path.join(FAISS_DIR, 'resnet_meta.pkl')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True) # Flask usually handles this

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# --- End Define Project Paths ---

# --- Load Model, Index, Metadata at Startup ---
logger.info("Loading ResNet50 model...")
try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    logger.info("ResNet50 model loaded.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load ResNet50 model: {e}", exc_info=True)
    exit() # Can't run without the model

logger.info(f"Loading FAISS index from: {INDEX_PATH}")
try:
    faiss_index = faiss.read_index(INDEX_PATH)
    logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load FAISS index: {e}", exc_info=True)
    logger.error("Ensure build_index.py ran successfully.")
    exit() # Can't run without the index

logger.info(f"Loading metadata from: {META_PATH}")
try:
    with open(META_PATH, 'rb') as f:
        meta_data = pickle.load(f) # Using pickle based on previous scripts
    logger.info(f"Metadata loaded for {len(meta_data)} images.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load metadata: {e}", exc_info=True)
    logger.error("Ensure extract_features.py ran successfully.")
    exit() # Can't run without metadata
# --- End Load Model, Index, Metadata ---

# --- Copy Feature Extraction Function (from scripts/extract_features.py) ---
def extract_resnet50_features(img_path):
    global resnet_model # Use the globally loaded model
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = resnet_model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        logger.error(f"Error extracting features for {img_path}: {e}", exc_info=True)
        return None
# --- End Copy Feature Extraction Function ---

# --- Copy Search Function (from scripts/search.py) ---
def search_similar_images(query_image_path, index, meta_data_dict, top_k=10):
    logger.info(f"Processing query image: {query_image_path}")
    if not os.path.exists(query_image_path):
        logger.error(f"Query image not found: {query_image_path}")
        return None

    features = extract_resnet50_features(query_image_path)
    if features is None:
        logger.error("Feature extraction failed for query image.")
        return None

    features = features.reshape(1, -1).astype('float32')

    logger.info(f"Searching for top {top_k} similar images...")
    try:
        distances, indices = index.search(features, top_k)
        similar_images_result = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                 # Check if index from FAISS exists in our metadata keys
                 if idx in meta_data_dict:
                      # Append tuple (path, score)
                      similar_images_result.append((meta_data_dict[idx], float(distances[0][i]))) # Ensure score is float
                 else:
                      logger.warning(f"Index {idx} from FAISS search result not found in metadata keys.")
        else:
             logger.warning("FAISS search returned no indices.")

        logger.info(f"Found {len(similar_images_result)} similar images.")
        return similar_images_result

    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return None
# --- End Copy Search Function ---

# --- CORRECTED Route to Serve Data Files (for results) ---
@app.route('/data/<path:filename>')
def serve_data(filename):
    logger.debug(f"--- Serving Data Request ---")
    logger.debug(f"Raw filename from URL: {filename}")
    # data_dir is the absolute path to the main 'data' directory
    data_dir = os.path.abspath(os.path.join(app.root_path, '..', 'data'))
    logger.debug(f"Serving from base directory: {data_dir}")
    try:
        # send_from_directory expects the base directory and the relative path within it
        return send_from_directory(data_dir, filename, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"File not found by send_from_directory: {filename} in {data_dir}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
         logger.error(f"Error serving file {filename}: {e}", exc_info=True)
         return jsonify({"error": "Server error"}), 500
    finally:
         logger.debug(f"--- End Serving Data Request ---")
# --- End Add Route ---


@app.route('/search', methods=['POST'])
def search():
    logger.info("Received request to /search.")
    socketio.emit('log_message', {'message': 'Received search request.'})

    if 'file' not in request.files:
        logger.error("No file part in the request.")
        socketio.emit('log_message', {'message': 'Error: No file part in the request.'})
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file.")
        socketio.emit('log_message', {'message': 'Error: No selected file.'})
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    # Save upload within the UPLOAD_FOLDER defined relative to project root
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        logger.info(f"Upload saved successfully: {filepath}")
        socketio.emit('log_message', {'message': f"File saved: {filename}"})
    except Exception as e:
        logger.error(f"Error saving file: {e}", exc_info=True)
        socketio.emit('log_message', {'message': f"Error saving file: {e}"})
        return jsonify({"error": f"Error saving file: {e}"}), 500

    # --- Perform the search ---
    socketio.emit('log_message', {'message': 'Starting similarity search...'})
    # Use the globally loaded index and metadata
    similar_images = search_similar_images(filepath, faiss_index, meta_data)
    # --- End Perform the search ---

    # --- Send results back via SocketIO ---
    if similar_images is not None:
        # Convert absolute server paths to relative URLs
        result_urls = []
        logger.debug("--- Debugging URL Conversion ---")
        for img_path, score in similar_images:
             # Make path relative to PROJECT_ROOT
             try:
                 logger.debug(f"Original img_path: {img_path}")
                 relative_path = os.path.relpath(img_path, PROJECT_ROOT)
                 logger.debug(f"Calculated relative_path: {relative_path}")
                 # --- CORRECTED URL PATH ---
                 # Prepend '/' only, as relative_path should start with 'data/...'
                 url_path = '/' + relative_path.replace(os.sep, '/')
                 # --- End CORRECTION ---
                 logger.debug(f"Final url_path: {url_path}")
                 result_urls.append({'url': url_path, 'score': score})
             except ValueError:
                  logger.warning(f"Could not make path relative; skipping: {img_path}")
        logger.debug("--- End Debugging URL Conversion ---")


        logger.info(f"Sending {len(result_urls)} results to client.")
        socketio.emit('search_results', {'results': result_urls})
        socketio.emit('log_message', {'message': f'Search complete. Found {len(result_urls)} similar images.'})
        # Return a simple success message; results are sent via SocketIO
        return jsonify({"message": "Search processing started and results sent via SocketIO."}), 200
    else:
        logger.error("Search failed or returned no results.")
        socketio.emit('log_message', {'message': 'Search failed or no results found.'})
        socketio.emit('search_results', {'results': []}) # Send empty results
        return jsonify({"error": "Search failed or no results found"}), 500
    # --- End Send results back ---


@app.route('/')
def home():
    # Serve site.html from the same directory as server.py (app)
    return send_from_directory(app.root_path, "site.html")

if __name__ == "__main__":
    logger.info("Starting Flask-SocketIO server...")
    # Removed invalid log_level argument
    socketio.run(app, debug=True, port=5000)