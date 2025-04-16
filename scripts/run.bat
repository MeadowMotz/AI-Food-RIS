py -3.10 -m pip install -r ../requirements.txt
py -3.10 download_dataset.py
py -3.10 preprocess.py
py -3.10 extract_features.py
py -3.10 build_index.py
pause