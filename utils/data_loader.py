import os
import pandas as pd
import torch
import requests
import zipfile
from typing import Dict, List, Tuple

def download_movielens(data_path: str):
    """Download and extract MovieLens data using Python-native methods"""
    ml_url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(data_path, "ml-100k.zip")
    
    # Download the zip file
    response = requests.get(ml_url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    # Remove the zip file
    os.remove(zip_path)

def load_movielens_data(data_path: str, min_rating: float) -> Dict:
    """Load MovieLens data with automatic download if needed"""
    # Create directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Check if data exists in Kaggle's default location first
    data_file = "/kaggle/input/movielens-100k-dataset/ml-100k/u.data"
    
    if not os.path.exists(data_file):
        # Fallback to manual download
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    elif not os.path.exists(f"{data_path}/u.data"):
        # Download if not present
        download_movielens(data_path)
        data_file = f"{data_path}/u.data"
    else:
        data_file = f"{data_path}/u.data"
    
    # Load data (standard MovieLens 100k format)
    ratings = pd.read_csv(
        data_file,
        sep="\t", 
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    
    # Filter by rating and preprocess
    ratings = ratings[ratings["rating"] >= min_rating]
    
    # Generate mock content features (replace with real features if available)
    unique_items = ratings["item_id"].unique()
    content_features = {
        iid: torch.randn(10)  # Mock 10-dim feature vector
        for iid in unique_items
    }
    
    # Split train/test
    test_mask = ratings["timestamp"] % 5 == 0  # 20% test split
    return {
        "user_ids": ratings["user_id"].unique().tolist(),
        "item_ids": ratings["item_id"].unique().tolist(),
        "item_features": content_features,
        "train_interactions": [
            (row.user_id, row.item_id, row.rating)
            for row in ratings[~test_mask].itertuples()
        ],
        "test_interactions": [
            (row.user_id, row.item_id, row.rating)
            for row in ratings[test_mask].itertuples()
        ]
    }