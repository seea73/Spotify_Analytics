"""Module for loading data from Kaggle API."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle
import pandas as pd
from pathlib import Path

from src.config import KAGGLE_DATASET, DATASET_FILE, BRONZE_DIR

def download_spotify_dataset():
    """Download Spotify dataset from Kaggle"""
    import os
    
    print(f"Downloading dataset {KAGGLE_DATASET}...")
    print(f"Dataset URL: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    
    
    os.makedirs(BRONZE_DIR, exist_ok=True)
    
    # Download the dataset
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=BRONZE_DIR,
        unzip=True,  
        quiet=False  
    )
    
   
    files = os.listdir(BRONZE_DIR)
    print(f"Files in {BRONZE_DIR}: {files}")
    
   
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        dataset_path = os.path.join(BRONZE_DIR, csv_files[0])
        print(f"Found CSV file: {dataset_path}")
        return dataset_path
    else:
        raise FileNotFoundError(f"No CSV files found in {BRONZE_DIR}")


def load_spotify_data():
    """Load the Spotify dataset from the bronze layer"""
    import pandas as pd
    import os
    from src.config import BRONZE_DIR
    
    # Path to the dataset
    file_path = os.path.join(BRONZE_DIR, 'spotify_tracks.csv')
    
   
    if not os.path.exists(file_path):
        file_path = download_spotify_dataset()  
    
    # Load the data
    df = pd.read_csv(file_path)
    
    return df