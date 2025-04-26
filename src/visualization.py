"""Module for creating visualizations from gold layer data."""
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import json
import redis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import VISUALIZATION_DIR, RedisConfig

class GoldLayerVisualizer:
    def __init__(self, redis_client=None):
        self.redis = redis_client or redis.Redis(
            host=RedisConfig.host,
            port=RedisConfig.port,
            db=RedisConfig.db,
            decode_responses=True
        )
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        sns.set(style="whitegrid")

    def _fetch_data(self, pattern):
        keys = self.redis.keys(pattern)
        data = []
        for key in keys:
            try:
                raw = self.redis.get(key)
                if raw:
                    data.append(json.loads(raw))
            except json.JSONDecodeError:
                print(f"Skipping bad JSON: {key}")
        return pd.DataFrame(data)

    def _save_fig(self, fig, name):
        path = os.path.join(VISUALIZATION_DIR, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")

    def plot_artist_popularity(self):
        df = self._fetch_data("spotify:gold:artist_popularity:*")
        if df.empty:
            return None

        df = df.sort_values("track_count", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x="track_count", y="artists", data=df, ax=ax, palette="magma")
        ax.set(title="Top 15 Artists by Track Count", xlabel="Track Count", ylabel="Artist")
        self._save_fig(fig, "artist_popularity")
        return fig

    def plot_yearly_trends(self):
        df = self._fetch_data("spotify:gold:year_trends:*")
        if df.empty or "release_year" not in df.columns:
            return None

        df = df.sort_values("release_year")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["release_year"], df["track_count"], marker="o", linestyle="-", linewidth=2)
        ax.set(title="Track Releases by Year", xlabel="Year", ylabel="Tracks Released")
        self._save_fig(fig, "yearly_trends")
        return fig

   

    def plot_popularity_segments(self):
        df = self._fetch_data("spotify:gold:popularity_segments:*")
        if df.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="popularity_segment", y="track_count", data=df, ax=ax, palette="coolwarm")
        ax.set(title="Tracks by Popularity Segment", xlabel="Segment", ylabel="Track Count")
        self._save_fig(fig, "popularity_segments")
        return fig

    def plot_audio_clusters(self):
        df = self._fetch_data("spotify:gold:audio_clusters:track_clusters:*")
        if df.empty or "cluster" not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x="cluster", palette="cubehelix", ax=ax)
        ax.set(title="Tracks per Audio Feature Cluster", xlabel="Cluster", ylabel="Count")
        self._save_fig(fig, "audio_clusters")
        return fig
    
   
    def plot_cluster_profiles(self):
        df = self._fetch_data("spotify:gold:audio_clusters:cluster_profile:*")
        if df.empty or 'cluster' not in df.columns:
            return None

        df = df.set_index("cluster").drop(columns="cluster_size", errors="ignore")
        df = df.transpose()

        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, colormap='tab20')
        ax.set(title="Audio Feature Averages by Cluster", ylabel="Feature Value", xlabel="Audio Feature")
        plt.xticks(rotation=45)
        self._save_fig(fig, "cluster_profiles")
        return fig

    
    
    

    def plot_popularity_segment_pie(self):
        df = self._fetch_data("spotify:gold:popularity_segments:*")
        if df.empty:
            return None

        fig, ax = plt.subplots()
        ax.pie(df["track_count"], labels=df["popularity_segment"], autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
        ax.axis("equal")
        ax.set_title("Popularity Segment Distribution")
        self._save_fig(fig, "popularity_segment_pie")
        return fig

    

    def plot_cluster_by_popularity_segment(self):
        df = self._fetch_data("spotify:gold:audio_clusters:track_clusters:*")
        if df.empty or not {"cluster", "popularity"}.issubset(df.columns):
            return None

        # Re-bin popularity into segments
        df["popularity_segment"] = pd.cut(
            df["popularity"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )

        pivot = df.pivot_table(index="popularity_segment", columns="cluster", aggfunc="size", fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set(title="Cluster Distribution by Popularity Segment", xlabel="Popularity Segment", ylabel="Track Count")
        plt.xticks(rotation=0)
        self._save_fig(fig, "cluster_by_popularity_segment")
        return fig

    def plot_cluster_centers_boxplots(self):
        df = self._fetch_data("spotify:gold:audio_clusters:cluster_centers:*")
        if df.empty:
            return None

        # Convert cluster centers to long format for box plotting
        df_long = df.melt(id_vars="cluster", var_name="feature", value_name="value")

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(data=df_long, x="feature", y="value", palette="Set3")
        ax.set(title="Distribution of Audio Features in Cluster Centers", xlabel="Audio Feature", ylabel="Feature Value")
        plt.xticks(rotation=45)
        self._save_fig(fig, "audio_cluster_centers_boxplot")
        return fig
    

    def plot_artist_track_clusters(self, n_clusters=5):
        df = self._fetch_data("spotify:gold:artist_popularity:*")
        if df.empty or "track_count" not in df.columns:
            return None

        # Prepare data
        artist_df = df[["artists", "track_count"]].copy()
        artist_df = artist_df.dropna().reset_index(drop=True)

        if len(artist_df) < n_clusters:
            print("Not enough artists for clustering.")
            return None

        # Scale for clustering
        scaler = StandardScaler()
        artist_df["scaled_count"] = scaler.fit_transform(artist_df[["track_count"]])

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        artist_df["cluster"] = kmeans.fit_predict(artist_df[["scaled_count"]])

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.stripplot(data=artist_df, x="cluster", y="track_count", jitter=True, palette="tab10", ax=ax)
        ax.set(title=f"Artists Grouped by Track Count ({n_clusters} Clusters)", xlabel="Cluster", ylabel="Track Count")
        self._save_fig(fig, "artist_track_clusters")
        return fig


    def generate_all(self):
        print("Generating visualizations (based on available Redis data)...")

        plots = {
            "artist_popularity": self.plot_artist_popularity(),
            "yearly_trends": self.plot_yearly_trends(),
            
            "popularity_segments": self.plot_popularity_segments(),
            "audio_clusters": self.plot_audio_clusters(),
            
            "cluster_profiles": self.plot_cluster_profiles(),
            
            
            "popularity_segment_pie": self.plot_popularity_segment_pie(),
           
            "cluster_by_popularity_segment": self.plot_cluster_by_popularity_segment(),
            "audio_cluster_centers_boxplot": self.plot_cluster_centers_boxplots(),
            "artist_track_clusters": self.plot_artist_track_clusters(n_clusters=5)


        }

        generated = [name for name, fig in plots.items() if fig is not None]
        skipped = [name for name in plots if plots[name] is None]

        print("\n✅ Generated:", ", ".join(generated) if generated else "None")
        print("❌ Skipped (no data):", ", ".join(skipped) if skipped else "None")


if __name__ == "__main__":
    visualizer = GoldLayerVisualizer()
    visualizer.generate_all()
