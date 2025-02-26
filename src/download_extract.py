import os

def ensure_data_directory(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please provide the dataset manually.")
