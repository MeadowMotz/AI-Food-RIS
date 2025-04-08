import os
import subprocess
import logging
import traceback
from flask import Flask, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from nbformat import read, write  
from werkzeug.utils import secure_filename
# from TODO import search_similar_images

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "../uploads"
STATIC_FOLDER = "../"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/search', methods=['POST'])
def search():
    logger.info("Received request to generate image.")
    socketio.emit('log_message', {'message': 'Received request to generate image.'})

    if 'file' not in request.files:
        logger.error("No file part in the request.")
        socketio.emit('log_message', {'message': 'No file part in the request.'})
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file.")
        socketio.emit('log_message', {'message': 'No selected file.'})
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    folder = os.path.dirname(file.filename).split(os.path.sep)[0] 
    if folder:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], folder, filename)
    else:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        logger.info(f"Upload saved successfully: {filepath}")
        socketio.emit('log_message', {'message': f"File saved: {filepath}"})
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        socketio.emit('log_message', {'message': f"Error saving file: {e}"})
        return jsonify({"error": f"Error saving file: {e}"}), 500

    # Run search script TODO
    logger.info("Beginning search...")
    socketio.emit('log_message', {'message': 'Beginning search...'})
    # result = search_similar_images(filepath)
    # convert to matlike? 

    if result is None:
        # After the notebook finishes execution, return the URL to the image
        image_url = STATIC_FOLDER + "output.png"
        socketio.emit('log_message', {'message': 'Executed successfully!'})
        return jsonify({"message": "Execution completed.", "image_url": image_url}), 200
    else:
        socketio.emit('log_message', {'message': f"Execution failed: {result}"})
        return jsonify({"error": result}), 500

@app.route('/')
def home():
    return send_file("site.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
