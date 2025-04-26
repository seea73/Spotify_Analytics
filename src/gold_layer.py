"""Gold layer: Create aggregated datasets for analysis."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import redis
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, GOLD_DIR
from src.silver_layer import connect_to_redis, fetch_all_data_from_redis


def create_artist_popularity_aggregation(df):
    print("Creating artist popularity aggregation...")
    
    artist_df = df.explode('artists').reset_index(drop=True)
    
    # Group by artist and calculate aggregations
    artist_agg = artist_df.groupby('artists').agg(
        track_count=('id', 'count'),
        avg_popularity=('popularity', 'mean'),
        avg_duration=('duration_min', 'mean'),
        avg_danceability=('danceability', 'mean'),
        avg_energy=('energy', 'mean'),
        avg_acousticness=('acousticness', 'mean'),
        explicit_track_ratio=('explicit', 'mean'),
        earliest_release=('release_date', 'min'),
        latest_release=('release_date', 'max')
    ).reset_index()
    
  
    artist_agg = artist_agg.sort_values('track_count', ascending=False)
    
 
    artist_agg = artist_agg[artist_agg['artists'].notna() & (artist_agg['artists'] != '')]
    
    print(f"Created artist aggregation with shape: {artist_agg.shape}")
    return artist_agg


def create_year_aggregation(df):
   
    print("Creating year aggregation...")
    
    # Filter out rows with missing release year
    year_df = df[df['release_year'].notna()].copy()
    
    # Group by release year and calculate aggregations
    year_agg = year_df.groupby('release_year').agg(
        track_count=('id', 'count'),
        avg_popularity=('popularity', 'mean'),
        avg_duration=('duration_min', 'mean'),
        avg_danceability=('danceability', 'mean'),
        avg_energy=('energy', 'mean'),
        avg_acousticness=('acousticness', 'mean'),
        avg_valence=('valence', 'mean'),
        explicit_ratio=('explicit', 'mean')
    ).reset_index()
    
    # Convert year to integer
    year_agg['release_year'] = year_agg['release_year'].astype(int)
    
    # Sort by year
    year_agg = year_agg.sort_values('release_year')
    
    print(f"Created year aggregation with shape: {year_agg.shape}")
    return year_agg


def create_audio_feature_clusters(df):
    print("Creating audio feature clusters...")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Select audio features
    features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    
    feature_df = df.dropna(subset=features).copy()
    
    if len(feature_df) < 1000:
        print("Not enough data for clustering after removing NaN values.")
        return None
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[features])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    feature_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Calculate cluster characteristics
    cluster_profile = feature_df.groupby('cluster').agg(
        cluster_size=('id', 'count'),
        avg_popularity=('popularity', 'mean'),
        avg_danceability=('danceability', 'mean'),
        avg_energy=('energy', 'mean'),
        avg_acousticness=('acousticness', 'mean'),
        avg_valence=('valence', 'mean'),
        avg_tempo=('tempo', 'mean')
    ).reset_index()
    
    print(f"Created audio feature clusters with shape: {cluster_profile.shape}")
    
    # Add cluster centers
    centers = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=features
    )
    centers['cluster'] = centers.index
    
    cluster_data = {
        'track_clusters': feature_df[['id', 'name', 'artists', 'popularity', 'cluster']],
        'cluster_profile': cluster_profile,
        'cluster_centers': centers
    }
    
    return cluster_data


def save_gold_data(data_dict, base_path=None):
    if base_path is None:
        base_path = GOLD_DIR
    
    paths = {}
    
    for name, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            file_path = base_path / f"{name}.csv"
            print(f"Saving {name} to {file_path}...")
            data.to_csv(file_path, index=False)
            paths[name] = file_path
        elif isinstance(data, dict):
            
            for sub_name, sub_data in data.items():
                if isinstance(sub_data, pd.DataFrame):
                    file_path = base_path / f"{name}_{sub_name}.csv"
                    print(f"Saving {name}_{sub_name} to {file_path}...")
                    sub_data.to_csv(file_path, index=False)
                    paths[f"{name}_{sub_name}"] = file_path
    
    return paths


def ingest_gold_data_to_redis(data_dict, client=None):
    if client is None:
        client = connect_to_redis()
    
    result = {}
    
    for name, data in data_dict.items():
        prefix = f"spotify:gold:{name}:"
        
        if isinstance(data, pd.DataFrame):
            
            existing_keys = client.keys(f"{prefix}*")
            if existing_keys:
                for key in existing_keys:
                    client.delete(key)
            
            print(f"Ingesting {len(data)} records for {name}...")
            
           
            if 'id' in data.columns:
                key_col = 'id'
           
            else:
                key_col = data.columns[0]
            
            pipe = client.pipeline()
            
            for idx, row in data.iterrows():
                record = row.to_dict()
                
                # Convert special types for JSON serialization
                for k, v in record.items():
                    if isinstance(v, pd.Timestamp):
                        record[k] = v.isoformat()
                    elif isinstance(v, np.int64):
                        record[k] = int(v)
                    elif isinstance(v, np.float64):
                        record[k] = float(v)
                
                key = f"{prefix}{record[key_col]}"
                pipe.set(key, json.dumps(record))
                
                if idx % 1000 == 0:
                    pipe.execute()
            
            pipe.execute()
            
            count = len(client.keys(f"{prefix}*"))
            print(f"Ingested {count} records for {name}.")
            result[name] = count
        
        elif isinstance(data, dict):
            # Handle nested dictionaries
            for sub_name, sub_data in data.items():
                if isinstance(sub_data, pd.DataFrame):
                    sub_prefix = f"spotify:gold:{name}:{sub_name}:"
                    
                    # Clear existing keys
                    existing_keys = client.keys(f"{sub_prefix}*")
                    if existing_keys:
                        for key in existing_keys:
                            client.delete(key)
                    
                    print(f"Ingesting {len(sub_data)} records for {name}_{sub_name}...")
                    
                    # Determine key column
                    if 'id' in sub_data.columns:
                        key_col = 'id'
                    else:
                        key_col = sub_data.columns[0]
                    
                    pipe = client.pipeline()
                    
                    for idx, row in sub_data.iterrows():
                        record = row.to_dict()
                        
                        # Convert special types
                        for k, v in record.items():
                            if isinstance(v, (pd.Timestamp, np.int64, np.float64)):
                                if isinstance(v, pd.Timestamp):
                                    record[k] = v.isoformat()
                                else:
                                    record[k] = float(v) if isinstance(v, np.float64) else int(v)
                        
                        if key_col in record:
                            key = f"{sub_prefix}{record[key_col]}"
                        else:
                            key = f"{sub_prefix}{idx}"
                        
                        pipe.set(key, json.dumps(record))
                        
                        if idx % 1000 == 0:
                            pipe.execute()
                    
                    pipe.execute()
                    
                    count = len(client.keys(f"{sub_prefix}*"))
                    print(f"Ingested {count} records for {name}_{sub_name}.")
                    result[f"{name}_{sub_name}"] = count
    
    return result


def create_genre_aggregation(df):
    print("Creating genre aggregation...")
    
    
    if 'genres' not in df.columns:
        print("No genres column found in data.")
        return None
    
    
    genre_df = df.explode('genres').reset_index(drop=True)
    genre_df = genre_df[genre_df['genres'].notna() & (genre_df['genres'] != '')]
    
    # Group by genre and calculate aggregations
    genre_agg = genre_df.groupby('genres').agg(
        track_count=('id', 'count'),
        avg_popularity=('popularity', 'mean'),
        avg_danceability=('danceability', 'mean'),
        avg_energy=('energy', 'mean'),
        avg_acousticness=('acousticness', 'mean'),
        avg_valence=('valence', 'mean'),
        avg_tempo=('tempo', 'mean')
    ).reset_index()
    
    # Sort by track count (descending)
    genre_agg = genre_agg.sort_values('track_count', ascending=False)
    
    print(f"Created genre aggregation with shape: {genre_agg.shape}")
    return genre_agg


def create_popularity_segments(df):
    
    print("Creating popularity segments...")
    
    # Create popularity segments
    df['popularity_segment'] = pd.cut(
        df['popularity'], 
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Group by popularity segment
    pop_segments = df.groupby('popularity_segment').agg(
        track_count=('id', 'count'),
        avg_danceability=('danceability', 'mean'),
        avg_energy=('energy', 'mean'),
        avg_acousticness=('acousticness', 'mean'),
        avg_valence=('valence', 'mean'),
        avg_tempo=('tempo', 'mean'),
        explicit_ratio=('explicit', 'mean')
    ).reset_index()
    
    print(f"Created popularity segments with shape: {pop_segments.shape}")
    return pop_segments


def create_track_recommendations(df):
    print("Creating track recommendations...")
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    
    # Select audio features for similarity
    features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Get complete records
    complete_df = df.dropna(subset=features).copy()
    
    if len(complete_df) < 100:
        print("Not enough complete data for recommendations.")
        return None
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(complete_df[features])
    
    # Create nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=6, algorithm='auto')
    nn_model.fit(scaled_features)
    
    # Find recommendations for top 1000 tracks
    top_tracks = complete_df.sort_values('popularity', ascending=False).head(1000)
    recommendations = {}
    
    for idx, track in top_tracks.iterrows():
        track_features = track[features].values.reshape(1, -1)
        scaled_track = scaler.transform(track_features)
        
        # Get nearest neighbors
        distances, indices = nn_model.kneighbors(scaled_track)
        
        # Skip the first result (which is the track itself)
        similar_tracks = complete_df.iloc[indices[0][1:], :][['id', 'name', 'artists']]
        
        recommendations[track['id']] = {
            'track_id': track['id'],
            'track_name': track['name'],
            'track_artists': track['artists'],
            'similar_tracks': similar_tracks.to_dict('records')
        }
    
    print(f"Created recommendations for {len(recommendations)} tracks")
    return recommendations


if __name__ == "__main__":
    # Test the gold layer
    client = connect_to_redis()
    clean_df = fetch_all_data_from_redis(client=client, prefix="spotify:clean:")
    
    if len(clean_df) == 0:
        print("No clean data found in Redis. Running silver layer first...")
        from src.silver_layer import fetch_all_data_from_redis as fetch_raw
        from src.silver_layer import clean_spotify_data, ingest_clean_data_to_redis
        
        raw_df = fetch_raw(client)
        clean_df = clean_spotify_data(raw_df)
        ingest_clean_data_to_redis(clean_df, client=client)
    
    # Create gold layer aggregations
    artist_agg = create_artist_popularity_aggregation(clean_df)
    year_agg = create_year_aggregation(clean_df)
    clusters = create_audio_feature_clusters(clean_df)
    genre_agg = create_genre_aggregation(clean_df)
    pop_segments = create_popularity_segments(clean_df)
    recommendations = create_track_recommendations(clean_df)
    
    # Create a dictionary of all gold layer data
    gold_data = {
        'artist_popularity': artist_agg,
        'year_trends': year_agg,
        'audio_clusters': clusters,
        'genre_analysis': genre_agg,
        'popularity_segments': pop_segments,
        'track_recommendations': recommendations
    }
    
    # Save to files
    file_paths = save_gold_data(gold_data)
    print(f"Saved gold layer data to {len(file_paths)} files.")
    
    # Ingest to Redis
    ingest_results = ingest_gold_data_to_redis(gold_data, client=client)
    print(f"Ingested {sum(ingest_results.values())} total records to Redis.")
    
    print("Gold layer processing complete!")