"""Bronze layer: Load raw data into Redis."""
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import redis
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from datetime import datetime

from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PREFIX
from src.data_loader import load_spotify_data

from pathlib import Path

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True) 


def connect_to_redis():
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True  
    )
    
    try:
        client.ping()
        print("Connected to Redis server successfully.")
        return client
    except redis.ConnectionError:
        print("Failed to connect to Redis server. Make sure Redis is running.")
        raise


def ingest_data_to_redis(df, client=None):
    if client is None:
        client = connect_to_redis()
    
    # Clear existing keys with the same prefix
    existing_keys = client.keys(f"{REDIS_PREFIX}*")
    if existing_keys:
        print(f"Clearing {len(existing_keys)} existing keys...")
        for key in existing_keys:
            client.delete(key)
    
    print(f"Ingesting {len(df)} records into Redis...")
    
    # Create a pipeline for bulk insertion
    pipe = client.pipeline()
    
    # Convert dataframe to JSON records and store in Redis
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Convert row to dictionary
        record = row.to_dict()
        
        # Create a unique key for each track
        key = f"{REDIS_PREFIX}{record['track_id']}"
        
        # Store as JSON string
        pipe.set(key, json.dumps(record))
        
        # Execute pipeline in batches to avoid memory issues
        if idx % 1000 == 0:
            pipe.execute()
    
    # Execute any remaining commands
    pipe.execute()
    
    # Verify the count of records
    count = len(client.keys(f"{REDIS_PREFIX}*"))
    print(f"Successfully ingested {count} records into Redis.")
    
    return count


def count_redis_records(client=None):
    if client is None:
        client = connect_to_redis()
    
    count = len(client.keys(f"{REDIS_PREFIX}*"))
    print(f"Found {count} records in Redis with prefix '{REDIS_PREFIX}'.")
    
    return count


def sample_redis_data(n=5, client=None):
    if client is None:
        client = connect_to_redis()
    
    keys = client.keys(f"{REDIS_PREFIX}*")
    
    if not keys:
        print("No data found in Redis.")
        return []
    
    sample_keys = keys[:min(n, len(keys))]
    samples = []
    
    for key in sample_keys:
        data = client.get(key)
        samples.append(json.loads(data))
    
    return samples


if __name__ == "__main__":
    # Test the bronze layer
    df = load_spotify_data()
    client = connect_to_redis()
    ingest_data_to_redis(df, client)
    count = count_redis_records(client)
    samples = sample_redis_data(5, client)
    
    print("\nSample data from Redis:")
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(json.dumps(sample, indent=2)[:500] + "...\n")

def fetch_all_data_from_redis(client=None, prefix=REDIS_PREFIX):
    if client is None:
        client = connect_to_redis()
    
    keys = client.keys(f"{prefix}*")
    
    if not keys:
        print(f"No data found in Redis with prefix '{prefix}'.")
        return pd.DataFrame()
    
    print(f"Fetching {len(keys)} records from Redis...")
    
    records = []
    for key in tqdm(keys):
        data = client.get(key)
        if data:
            try:
                record = json.loads(data)
                records.append(record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for key: {key}")
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    print(f"Created DataFrame with shape: {df.shape}")
    
    return df

def clean_spotify_data(df):
    
    print("Cleaning Spotify data...")
    
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Generate a unique ID if not present
    if 'id' not in clean_df.columns and 'track_id' in clean_df.columns:
        clean_df['id'] = clean_df['track_id']
    
    # Convert duration from ms to minutes
    if 'duration_ms' in clean_df.columns:
        clean_df['duration_min'] = clean_df['duration_ms'] / 60000
    
    # Parse artist strings to lists 
    if 'artists' in clean_df.columns and isinstance(clean_df['artists'].iloc[0], str):
        clean_df['artists'] = clean_df['artists'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    
    # Extract release year 
    if 'album_name' in clean_df.columns:
       
        clean_df['release_date'] = pd.to_datetime('2020-01-01')
        clean_df['release_year'] = 2020
    
    # Remove duplicates
    clean_df = clean_df.drop_duplicates(subset=['id'])
    
    # Rename track name column 
    if 'track_name' in clean_df.columns and 'name' not in clean_df.columns:
        clean_df['name'] = clean_df['track_name']
    
    print(f"Cleaned data shape: {clean_df.shape}")
    return clean_df

def save_clean_data(df, save_path=None):
    
    if save_path is None:
        save_path = SILVER_DIR / "spotify_tracks_clean.csv"
    
    print(f"Saving cleaned data to {save_path}...")
    df.to_csv(save_path, index=False)
    
    return save_path


def ingest_clean_data_to_redis(df, client=None):
    
    if client is None:
        client = connect_to_redis()
    
    clean_prefix = "spotify:clean:"
    
    # Clear existing keys with the clean prefix
    existing_keys = client.keys(f"{clean_prefix}*")
    if existing_keys:
        print(f"Clearing {len(existing_keys)} existing clean keys...")
        for key in existing_keys:
            client.delete(key)
    
    print(f"Ingesting {len(df)} cleaned records into Redis...")
    
    # Create a pipeline for bulk insertion
    pipe = client.pipeline()
    
    # Convert dataframe to JSON records and store in Redis
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Convert row to dictionary
        record = row.to_dict()
        
        # Handle non-serializable objects
        for k, v in record.items():
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                record[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
            elif isinstance(v, (np.int64, np.float64)):
                record[k] = int(v) if isinstance(v, np.int64) else float(v)
            elif isinstance(v, list) and all(isinstance(item, (np.int64, np.float64)) for item in v):
                record[k] = [int(item) if isinstance(item, np.int64) else float(item) for item in v]
        
        # Create a unique key for each track
        key = f"{clean_prefix}{record['id']}"
        
        # Store as JSON string
        pipe.set(key, json.dumps(record))
        
        # Execute pipeline in batches to avoid memory issues
        if idx % 1000 == 0:
            pipe.execute()
    
    
    pipe.execute()
    
    # Verify the count of records
    count = len(client.keys(f"{clean_prefix}*"))
    print(f"Successfully ingested {count} cleaned records into Redis.")
    
    return count

if __name__ == "__main__":
    # Test the silver layer
    client = connect_to_redis()
    raw_df = fetch_all_data_from_redis(client)
    clean_df = clean_spotify_data(raw_df)
    save_path = save_clean_data(clean_df)
    ingest_clean_data_to_redis(clean_df, client=client)