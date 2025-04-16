import os
import logging
import pickle
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
import faiss
from scripts.search import search_similar_images
import uvicorn
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FAISS_DIR = os.path.join(PROJECT_ROOT, "faiss")
INDEX_PATH = os.path.join(FAISS_DIR, 'food_faiss.index')
META_PATH = os.path.join(FAISS_DIR, 'resnet_meta.pkl')
SIMILAR_FOLDER = os.path.join(PROJECT_ROOT, "data", "similar")  # Directory to store search result file

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIMILAR_FOLDER, exist_ok=True)  # Ensure the folder for storing results exists

logger.info(f"Loading FAISS index from: {INDEX_PATH}")
try:
    faiss_index = faiss.read_index(INDEX_PATH)
    logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load FAISS index: {e}", exc_info=True)
    exit()

logger.info(f"Loading metadata from: {META_PATH}")
try:
    with open(META_PATH, 'rb') as f:
        meta_data = pickle.load(f)
    logger.info(f"Metadata loaded for {len(meta_data)} images.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load metadata: {e}", exc_info=True)
    exit()

@app.get("/data/{filename:path}")
def serve_data(filename: str):
    logger.debug(f"Serving file {filename}")
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/search")
async def search(file: UploadFile = File(...)):
    logger.info("Received request to /search.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File uploaded successfully: {file.filename}")
    except Exception as e:
        logger.error(f"Error saving file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    logger.info("Starting similarity search...")
    search_similar_images(file_path, faiss_index, meta_data)

    search_result_file = os.path.join(SIMILAR_FOLDER, "search_result.txt")
    logger.info(f"Checking {search_result_file}")
    if (os.path.exists(search_result_file)):
        # Now read the file contents and send them back as part of the response
        with open(search_result_file, 'r') as file:
            search_results = file.readlines()

        result_urls = []
        for line in search_results:
            img_path, score = line.strip().split(": ")

            # Get the parts of the path
            path_parts = img_path.split(os.sep)

            # Find the index where "data" occurs
            data_index = path_parts.index("data")

            # Everything after "data" is the portion we need
            relevant_parts = path_parts[data_index + 1:]  # Everything after "data"
            
            # Create the URL path by joining the relevant parts
            url_path = f"/data/{'/'.join(relevant_parts)}"

            result_urls.append({'url': url_path, 'score': float(score)})

        logger.info(f"Search complete. Found {len(result_urls)} similar images.")
        return {"message": "Search complete", "results": result_urls}
    else:
        logger.error("Search failed or returned no results.")
        raise HTTPException(status_code=500, detail="Search failed or no results found")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    file_path = os.path.join(PROJECT_ROOT, "app", "site.html")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Site HTML not found")

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="localhost", port=5000)
