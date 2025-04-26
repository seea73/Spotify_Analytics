"""Utility functions for the Spotify Redis Analytics project."""
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pandas as pd
import json
import redis
import matplotlib.pyplot as plt
from datetime import datetime
from src.config import RedisConfig

class SpotifyUtils:
    @staticmethod
    def ensure_directory(directory_path):
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
    
    @staticmethod
    def save_dataframe_to_csv(df, file_path, index=False):
        
        directory = os.path.dirname(file_path)
        SpotifyUtils.ensure_directory(directory)
        
        df.to_csv(file_path, index=index)
        print(f"Saved DataFrame to {file_path} ({len(df)} rows)")
    
    @staticmethod
    def save_dataframe_to_json(df, file_path):
        
        directory = os.path.dirname(file_path)
        SpotifyUtils.ensure_directory(directory)
        
        df.to_json(file_path, orient='records')
        print(f"Saved DataFrame to {file_path} ({len(df)} rows)")
    
    @staticmethod
    def save_figures(figures, directory_path):
        
        SpotifyUtils.ensure_directory(directory_path)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        for name, fig in figures.items():
            file_path = os.path.join(directory_path, f"{name}_{timestamp}.png")
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {file_path}")
    
    @staticmethod
    def get_redis_info():
       
        try:
            r = redis.Redis(
                host=RedisConfig.host,
                port=RedisConfig.port,
                db=RedisConfig.db,
                decode_responses=True
            )
            
            # Get key counts by layer
            bronze_count = len(r.keys("bronze:*"))
            silver_count = len(r.keys("silver:*"))
            gold_count = len(r.keys("gold:*"))
            model_count = len(r.keys("model:*"))
            
            # Get Redis server info
            info = r.info()
            
            return {
                "connected": True,
                "bronze_keys": bronze_count,
                "silver_keys": silver_count,
                "gold_keys": gold_count,
                "model_keys": model_count,
                "total_keys": bronze_count + silver_count + gold_count + model_count,
                "memory_used_human": f"{info.get('used_memory_human', 'N/A')}",
                "redis_version": info.get('redis_version', 'N/A'),
                "uptime_days": info.get('uptime_in_days', 'N/A')
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
        
    @staticmethod
    def get_redis_info():
        client = redis.Redis(
            host=RedisConfig.host,
            port=RedisConfig.port,
            db=RedisConfig.db,
            decode_responses=True
        )

        # Match actual key prefixes used during ingestion
        bronze_keys = len(client.keys("spotify:*"))                          # all raw keys
        silver_keys = len(client.keys("spotify:clean:*"))                    # cleaned tracks
        gold_keys = len(client.keys("spotify:gold:*"))                       # all gold aggregations
        model_keys = len(client.keys("*predictor.pkl")) + len(client.keys("*clusters.pkl"))

        total_keys = len(client.keys("*"))
        memory_used = client.info("memory").get("used_memory_human", "N/A")
        version = client.info("server").get("redis_version", "N/A")
        uptime_days = client.info("server").get("uptime_in_days", "N/A")

        return {
            "connected": True,
            "bronze_keys": bronze_keys,
            "silver_keys": silver_keys,
            "gold_keys": gold_keys,
            "model_keys": model_keys,
            "total_keys": total_keys,
            "memory_used_human": memory_used,
            "redis_version": version,
            "uptime_days": uptime_days
        }

    
    @staticmethod
    def extract_release_year(date_str):
        
        if not date_str or pd.isna(date_str):
            return None
        
        try:
            
            for fmt in ('%Y-%m-%d', '%Y-%m', '%Y', '%d-%m-%Y', '%m/%d/%Y'):
                try:
                    date_obj = datetime.strptime(str(date_str), fmt)
                    return date_obj.year
                except ValueError:
                    continue
            
            
            if isinstance(date_str, str) and len(date_str) >= 4:
                year_str = date_str[:4]
                if year_str.isdigit():
                    year = int(year_str)
                    if 1900 <= year <= datetime.now().year:
                        return year
            
            return None
        except Exception:
            return None
    
    @staticmethod
    def summarize_dataframe(df, name="DataFrame"):
       
        print(f"\n{'='*50}")
        print(f" {name} Summary ")
        print(f"{'='*50}")
        
        print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print("\nColumn Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
        
        print("\nMissing Values:")
        for col, missing in df.isna().sum().items():
            if missing > 0:
                pct = (missing / len(df)) * 100
                print(f"  - {col}: {missing} ({pct:.2f}%)")
        
        print("\nSample Data:")
        print(df.head(3))
        print(f"{'='*50}\n")
