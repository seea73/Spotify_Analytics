"""Bronze layer: Load raw data into Redis."""

import sys
import os
import json


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import redis
import pandas as pd
from tqdm import tqdm


from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PREFIX
from src.data_loader import load_spotify_data


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
    
   
    existing_keys = client.keys(f"{REDIS_PREFIX}*")
    if existing_keys:
        print(f"Clearing {len(existing_keys)} existing keys...")
        for key in existing_keys:
            client.delete(key)
    
    print(f"[DEBUG] Ingested {len(df)} rows into Redis (bronze)")

    
    # Create a pipeline for bulk insertion
    pipe = client.pipeline()
    
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        record = row.to_dict()
        
        # Create a unique key for each track
        key = f"{REDIS_PREFIX}{record['track_id']}"
        
        
        pipe.set(key, json.dumps(record))
        
        # Execute pipeline in batches to avoid memory issues
        if idx % 1000 == 0:
            pipe.execute()
    
    
    pipe.execute()
    
    
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
    
    df = load_spotify_data()
    client = connect_to_redis()
    ingest_data_to_redis(df, client)
    count = count_redis_records(client)
    samples = sample_redis_data(5, client)
    
    print("\nSample data from Redis:")
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(json.dumps(sample, indent=2)[:500] + "...\n")
