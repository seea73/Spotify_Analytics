import os
import logging
import argparse
from datetime import datetime

import redis

from src.config import RedisConfig, VISUALIZATION_DIR
from src.data_loader import download_spotify_dataset, load_spotify_data
from src.bronze_layer import ingest_data_to_redis
from src.silver_layer import fetch_all_data_from_redis, clean_spotify_data, ingest_clean_data_to_redis
from src.gold_layer import (
    create_artist_popularity_aggregation, create_year_aggregation, create_audio_feature_clusters,
    create_genre_aggregation, create_popularity_segments, create_track_recommendations,
    save_gold_data, ingest_gold_data_to_redis
)
from src.visualization import GoldLayerVisualizer
from src.ml_models import SpotifyMLModels
from src.utils import SpotifyUtils

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spotify_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_redis():
    logger.info("Connecting to Redis...")
    client = redis.Redis(
        host=RedisConfig.host,
        port=RedisConfig.port,
        db=RedisConfig.db,
        decode_responses=True
    )
    client.ping()
    logger.info("Connected to Redis.")
    return client

def run_pipeline(skip_download=False):
    start_time = datetime.now()
    logger.info("Starting Spotify Redis Analytics pipeline...")

    client = connect_to_redis()

    # Step 1: Download
    if not skip_download:
        download_spotify_dataset()

    # Step 2: Bronze layer
    df = load_spotify_data()
    print(f"[DEBUG] Raw data shape: {df.shape}")
    ingest_data_to_redis(df, client)
    print(f"[DEBUG] Ingested bronze layer")

    # Step 3: Silver layer
    raw_df = fetch_all_data_from_redis(client)
    print(f"[DEBUG] Fetched from Redis (bronze): {raw_df.shape}")
    clean_df = clean_spotify_data(raw_df)
    print(f"[DEBUG] Cleaned data shape: {clean_df.shape}")

    ingest_clean_data_to_redis(clean_df, client)
    print(f"[DEBUG] Cleaned data ingested to Redis.")
    
    os.makedirs("data/silver", exist_ok=True)
    clean_df.to_csv("data/silver/silver_dataset.csv", index=False)
    print("[✔] Saved cleaned silver data to data/silver/silver_dataset.csv")



    # Step 4: Gold layer
    logger.info("Generating gold layer data...")

    gold_data = {
        "artist_popularity": create_artist_popularity_aggregation(clean_df),
        "year_trends": create_year_aggregation(clean_df),
        "audio_clusters": create_audio_feature_clusters(clean_df),
        "popularity_segments": create_popularity_segments(clean_df),
        "track_recommendations": create_track_recommendations(clean_df)
    }

    for key, value in gold_data.items():
        if value is None:
            logger.warning(f"{key} returned None — check inputs or logic.")
        else:
            logger.info(f"{key}: generated with shape {value.shape if hasattr(value, 'shape') else 'N/A'}")

    logger.info("Saving gold data...")
    save_gold_data(gold_data)

    logger.info("Ingesting gold data into Redis...")
    ingest_gold_data_to_redis(gold_data, client)


    # Step 5: Visualizations
    logger.info("Generating visualizations...")
    visualizer = GoldLayerVisualizer(redis_client=client)
    visualizer.generate_all()

    

    # Step 6: Machine Learning
    logger.info("Running ML models...")
    ml = SpotifyMLModels(redis_client=client)
    cluster_result = ml.cluster_songs_by_audio_features()
    pop_model_result = ml.predict_song_popularity()

    logger.info("ML Model Results:")
    if cluster_result:
        logger.info(f"Clustering - Samples: {cluster_result['n_samples']}, Silhouette: {cluster_result['silhouette_score']:.2f}")
    else:
        logger.warning("Clustering model failed or returned None.")

    if pop_model_result:
        logger.info(f"Popularity Prediction - R²: {pop_model_result['r2']:.2f}, MSE: {pop_model_result['mse']:.2f}")
    else:
        logger.warning("Popularity prediction model failed or returned None.")


    # Step 7: Utils Summary
    logger.info("\n" + "="*50)
    logger.info("Redis Summary")
    logger.info("="*50)
    redis_info = SpotifyUtils.get_redis_info()
    for key, value in redis_info.items():
        logger.info(f"{key}: {value}")

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Pipeline finished in {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Redis Analytics Pipeline")
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download step')
    args = parser.parse_args()

    try:
        run_pipeline(skip_download=args.skip_download)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
