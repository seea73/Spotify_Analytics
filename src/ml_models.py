import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import redis
import json
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from src.config import RedisConfig
from src.config import MODEL_DIR


class SpotifyMLModels:
    def __init__(self, redis_client=None):
        
        if redis_client is None:
            self.redis = redis.Redis(
                host=RedisConfig.HOST,
                port=RedisConfig.PORT,
                password=RedisConfig.PASSWORD,
                decode_responses=True
            )
        else:
            self.redis = redis_client
    
    def _get_data_from_redis(self, key_pattern):
        """Retrieve data from Redis."""
        keys = self.redis.keys(key_pattern)
        data = []
        
        for key in keys:
            json_data = self.redis.get(key)
            if json_data:
                try:
                    data.append(json.loads(json_data))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for key: {key}")
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def _save_model_to_file(self, model_data, model_name):
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model_data, model_path)
        print(f"Model '{model_name}' saved to {model_path}")

    
    def cluster_songs_by_audio_features(self, n_clusters=5, save_model=True):
    
        # Get data from silver layer (cleaned data)
        df = self._get_data_from_redis("spotify:clean:*")
        
        if df.empty:
            print("No track data found in Redis")
            return None
        
        # Select features for clustering
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 
                          'valence', 'tempo']
        
        # Check which features are available
        available_features = [col for col in audio_features if col in df.columns]
        
        if not available_features:
            print("No audio features available for clustering")
            return None
        
        # Drop rows with missing values
        features_df = df[available_features].dropna()
        
        if len(features_df) < 100:
            print("Not enough data points for meaningful clustering")
            return None
            
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score for cluster quality
        silhouette = silhouette_score(scaled_features, clusters) if n_clusters > 1 else 0
        
        # Add cluster assignments back to original data
        result_df = features_df.copy()
        result_df['cluster'] = clusters
        
        # Calculate cluster profiles (mean of each feature per cluster)
        cluster_profiles = result_df.groupby('cluster').mean().reset_index()
        
        # Save model if requested
        if save_model:
            model_data = {
                'kmeans': kmeans,
                'scaler': scaler,
                'features': available_features
            }
            self._save_model_to_file(model_data, "kmeans_audio_clusters")
        
        # Return results
        return {
            'model': kmeans,
            'silhouette_score': silhouette,
            'cluster_profiles': cluster_profiles.to_dict(orient='records'),
            'n_samples': len(result_df)
        }
    
    def predict_song_popularity(self, save_model=True):
        # Get data from silver layer
        df = self._get_data_from_redis("spotify:clean:*")
        
        if df.empty:
            print("No track data found in Redis")
            return None
        
        # Check if popularity column exists
        if 'popularity' not in df.columns:
            print("Popularity column not found in data")
            return None
        
        # Select features for prediction
        features = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo', 'explicit', 'duration_ms']
        
        # Check which features are available
        available_features = [col for col in features if col in df.columns]
        
        if not available_features:
            print("No audio features available for prediction")
            return None
        
        # Prepare the data
        X = df[available_features].fillna(0)
        y = df['popularity']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model if requested
        if save_model:
            model_data = {
                'model': model,
                'features': available_features
            }
            self._save_model_to_file(model_data, "popularity_predictor")
        
        # Return results
        return {
            'model': model,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance.to_dict(orient='records'),
            'n_samples': len(X)
        }
    
    def recommend_similar_tracks(self, track_id=None, n_recommendations=5):
        # Get data from silver layer
        df = self._get_data_from_redis("silver:track:*")
        
        if df.empty:
            print("No track data found in Redis")
            return None
        
        # Select a random track if none specified
        if track_id is None or track_id not in df['id'].values:
            track_id = df['id'].sample(1).values[0]
            print(f"Using random track ID: {track_id}")
        
        # Get the reference track
        reference_track = df[df['id'] == track_id].iloc[0]
        
        # Select features for similarity calculation
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 
                          'valence', 'tempo']
        
        # Check which features are available
        available_features = [col for col in audio_features if col in df.columns]
        
        if not available_features:
            print("No audio features available for similarity calculation")
            return None
        
        # Scale the features
        scaler = StandardScaler()
        features_matrix = scaler.fit_transform(df[available_features])
        
        # Convert to DataFrame 
        scaled_df = pd.DataFrame(features_matrix, index=df.index, columns=available_features)
        
        # Get the reference track features (scaled)
        reference_features = scaled_df.loc[reference_track.name]
        
        # Calculate Euclidean distance for each track
        distances = []
        for idx, row in scaled_df.iterrows():
            if idx != reference_track.name:  # Skip the reference track itself
                distance = np.linalg.norm(reference_features - row)
                distances.append((idx, distance))
        
        # Sort by distance and get top N recommendations
        distances.sort(key=lambda x: x[1])
        recommendations = distances[:n_recommendations]
        
        # Return the recommended tracks
        result = []
        for idx, distance in recommendations:
            track = df.loc[idx]
            similarity = 1 / (1 + distance)  
            result.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artists', 'Unknown'),
                'similarity_score': similarity
            })
        
        return {
            'reference_track': {
                'id': reference_track['id'],
                'name': reference_track.get('name', 'Unknown'),
                'artists': reference_track.get('artists', 'Unknown')
            },
            'recommendations': result
        }