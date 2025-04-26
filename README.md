# Spotify Redis Analytics Project

This project analyzes Spotify tracks data using Redis as a distributed big data storage solution. It implements a complete data pipeline from raw data ingestion to visualization and machine learning models.

## Project Overview

This project demonstrates:
- Data ingestion from Kaggle API into Redis
- Data cleaning and transformation
- Data aggregation for analytics
- Data visualization
- Machine learning model development

## Project Structure

```
spotify-redis-analytics/
│
├── poetry.toml
├── pyproject.toml
├── README.md
|── main.py
│
├── data/
│   ├── bronze/   # Raw data
│   ├── silver/   # Cleaned data
│   └── gold/     # Aggregated data for analytics
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── src/
    ├── __init__.py
    ├── config.py           # Configuration settings
    ├── data_loader.py      # Kaggle API and data loading
    ├── bronze_layer.py     # Data ingestion to Redis
    ├── silver_layer.py     # Data cleaning
    ├── gold_layer.py       # Data aggregation
    ├── visualization.py    # Data visualization
    ├── ml_models.py        # Machine learning models
    └── utils.py            # Utility functions
```

## Tech Stack

- **Redis**: Used as the primary big data storage solution
- **Python**: Programming language for data processing and analysis
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models
- **Poetry**: Dependency management
- **Kaggle API**: Data source
