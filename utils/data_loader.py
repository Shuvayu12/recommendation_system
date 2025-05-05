import os
import pandas as pd
import torch
import requests
import zipfile
from typing import Dict


def download_movielens(data_path: str):
    """Download and extract MovieLens 100K dataset into `data_path`"""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(data_path, "ml-100k.zip")

    print(f"Downloading MovieLens data to {zip_path}...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    os.remove(zip_path)
    print("Done.")


def load_movielens_data(data_path: str, min_rating: float) -> Dict:
    """
    Load MovieLens data, automatically download if not available.
    Works both on Kaggle and local environments.
    """
    kaggle_data_file = "/kaggle/input/movielens-100k-dataset/ml-100k/u.data"
    local_data_file = os.path.join(data_path, "ml-100k", "u.data")

    # Create data directory if needed
    os.makedirs(data_path, exist_ok=True)

    if os.path.exists(kaggle_data_file):
        data_file = kaggle_data_file
    else:
        if not os.path.exists(local_data_file):
            download_movielens(data_path)
        data_file = local_data_file

    # Load ratings
    ratings = pd.read_csv(
        data_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )

    # Filter ratings
    ratings = ratings[ratings["rating"] >= min_rating]

    # Generate mock item features
    item_ids = ratings["item_id"].unique()
    item_features = {
        iid: torch.randn(10) for iid in item_ids
    }

    # Simple train/test split by timestamp
    test_mask = ratings["timestamp"] % 5 == 0
    train = ratings[~test_mask]
    test = ratings[test_mask]

    return {
        "user_ids": ratings["user_id"].unique().tolist(),
        "item_ids": item_ids.tolist(),
        "item_features": item_features,
        "train_interactions": [
            (row.user_id, row.item_id, row.rating) for row in train.itertuples()
        ],
        "test_interactions": [
            (row.user_id, row.item_id, row.rating) for row in test.itertuples()
        ],
    }
