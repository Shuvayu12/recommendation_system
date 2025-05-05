import os
import pandas as pd
import torch
from typing import Dict, List, Tuple

def load_movielens_data(data_path: str, min_rating: float) -> Dict:
    """Load MovieLens data directly in Kaggle (no local download needed)."""
    # Kaggle paths (MovieLens 100k is pre-available)
    if not os.path.exists(f"{data_path}/u.data"):
        # Download if not present (e.g., in Kaggle)
        os.makedirs(data_path, exist_ok=True)
        !wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O ml-100k.zip
        !unzip ml-100k.zip -d {data_path}/
    
    # Load data (standard MovieLens 100k format)
    ratings = pd.read_csv(
        f"{data_path}/u.data", 
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