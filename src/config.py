"""Configuration settings for the Spotify Redis Analytics project."""
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Create directories 
for dir_path in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"
DATASET_FILE = "spotify_tracks.csv"

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PREFIX = "spotify:"

class RedisConfig:
    
    host = REDIS_HOST
    port = REDIS_PORT
    db = REDIS_DB
    prefix = REDIS_PREFIX
    
    @classmethod
    def get_redis_url(cls):
        return f"redis://{cls.host}:{cls.port}/{cls.db}"
    

COLUMNS_TO_KEEP = [
    "id", "name", "popularity", "duration_ms", "explicit", 
    "artists", "release_date", "danceability", "energy",
    "key", "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo"
]

# Visualization settings
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"
VISUALIZATION_DIR.mkdir(exist_ok=True)

MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

@classmethod
def get_client(cls, decode_responses=True):
    return redis.Redis(
        host=cls.host,
        port=cls.port,
        db=cls.db,
        decode_responses=decode_responses
    )
