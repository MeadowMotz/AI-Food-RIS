from flask import Flask, request, send_file
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Check if file is uploaded
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)  # Save the uploaded image

    # Run the Jupyter notebook with the uploaded image file path
    result = subprocess.run(
    [
        "conda", "run", "--name", "tf_cpu", "jupyter", "nbconvert", "--to", "notebook", 
        "--execute", "TeamProjectCheckpoint2.ipynb", "--output", "output_notebook.ipynb", 
        "--ExecutePreprocessor.timeout=600", "--execute", "--allow-errors",
        "--ExecutePreprocessor.kernel_name='python'"
    ],
    capture_output=True, text=True)


    print(result.stdout)
    print(result.stderr)


    # Check for errors in the notebook execution
    if result.returncode != 0:
        return f"Error executing notebook: {result.stderr}", 500

    # Return the generated image
    output_image = "./uploads/output.png"
    if not os.path.exists(output_image):
        return "Image not generated", 404


    return send_file(output_image, mimetype='image/png')

@app.route('/')
def home():
    return send_file("site.html")  # Serve the HTML page

if __name__ == "__main__":
    app.run(debug=True, port=5000)
