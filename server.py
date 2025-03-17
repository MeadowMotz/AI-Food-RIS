import os
import subprocess
import logging
import traceback
from flask import Flask, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from nbformat import read, write  
from werkzeug.utils import secure_filename
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

def run_notebook(filepath):
    """
    Executes the Jupyter notebook and streams real-time logs.
    """
    try:
        socketio.emit('log_message', {'message': 'Executing Jupyter notebook...'})
        socketio.emit('progress_update', {'progress': 0})

        # Modify notebook to execute only cell 4
        with open("TeamProjectCheckpoint2.ipynb") as f:
            nb = read(f, as_version=4)
        
        logger.info("Commenting out cells except 4...")
        for i, cell in enumerate(nb.cells):
            if i != 2:  
                cell['source'] = '\n'.join([f"# {line}" for line in cell['source'].splitlines()])

        modified_notebook_path = "modified_notebook.ipynb"
        with open(modified_notebook_path, 'w') as f:
            write(nb, f)

        logger.info("Running Jupyter notebook with subprocess...")

        cmd = [
            "conda", "run", "-n", "tf_cpu", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", modified_notebook_path,
            "--output", "output_notebook.ipynb", "--ExecutePreprocessor.timeout=600",
            "--execute", "--allow-errors", "--ExecutePreprocessor.kernel_name=python"
        ]

        # Start the subprocess with unbuffered output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        # Stream stdout and stderr to logs and socket
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                socketio.emit('log_message', {'message': line})
                logger.info(line)

        for error_line in iter(process.stderr.readline, ''):
            error_line = error_line.strip()
            if error_line:
                logger.error(error_line)

        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()

        if return_code != 0:
            error_msg = f"Error executing Jupyter notebook. Return code: {return_code}"
            logger.error(error_msg)
            socketio.emit('log_message', {'message': error_msg})
            socketio.emit('completed', {'message': 'Notebook execution failed. Check logs for details.'})
            return error_msg

        socketio.emit('progress_update', {'progress': 100})
        socketio.emit('log_message', {'message': 'Jupyter notebook executed successfully.'})
        socketio.emit('completed', {'message': 'Notebook execution completed successfully! Closing log modal.'})

        # Move generated image to static folder
        output_image_path = os.path.join(STATIC_FOLDER, "output.png")

        # **CHECK IF FILE EXISTS BEFORE MOVING**
        if os.path.exists(output_image_path):
            logger.info(f"Image saved successfully: {output_image_path}")
            socketio.emit('log_message', {'message': f"Image saved: {output_image_path}"})
        else:
            logger.error("Error: output.png not found.")
            socketio.emit('log_message', {'message': "Error: output.png not found."})
            return "Error: output.png not found."  # Fix return type

        return None  # Success

    except Exception as e:
        error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg

@app.route('/generate_image', methods=['POST'])
def generate_image():
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

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    try:
        file.save(filepath)
        logger.info(f"File saved successfully: {filepath}")
        socketio.emit('log_message', {'message': f"File saved: {filepath}"})
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        socketio.emit('log_message', {'message': f"Error saving file: {e}"})
        return jsonify({"error": f"Error saving file: {e}"}), 500

    os.environ["IMAGE_PATH"] = filepath  # Pass the file path to the Jupyter Notebook

    # Run the notebook synchronously to ensure Flask waits for completion
    result = run_notebook(filepath)

    if result is None:
        # After the notebook finishes execution, return the URL to the image
        image_url = f"/static/images/output.png"
        socketio.emit('log_message', {'message': 'Notebook execution completed successfully!'})
        return jsonify({"message": "Notebook execution completed.", "image_url": image_url}), 200
    else:
        socketio.emit('log_message', {'message': f"Notebook execution failed: {result}"})
        return jsonify({"error": result}), 500

@app.route('/')
def home():
    return send_file("site.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
