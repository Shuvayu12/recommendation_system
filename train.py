import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List
import numpy as np
from datetime import datetime

from models.hybrid import HybridRecommender
from utils.data_loader import load_movielens_data
from utils.metrics import calculate_metrics
from config import config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np
from tqdm import tqdm

def train_recommender():
    # Load data
    data = load_movielens_data(config.data_path, config.min_rating)
    
    # Initialize recommender with device awareness
    recommender = HybridRecommender(config)
    recommender.initialize(
        user_ids=data['user_ids'],
        item_ids=data['item_ids'],
        item_content_features=data['item_features']
    )
    
    # Convert interactions to device
    train_interactions = [
        (recommender.user_map[u], recommender.item_map[i], r)
        for u, i, r in data['train_interactions']
        if u in recommender.user_map and i in recommender.item_map
    ]
    
    # Initial graph update with validation
    print("Building initial graph...")
    recommender.update_graph(data['train_interactions'])
    
    # Debug checks
    print(f"\nDevice Check:")
    print(f"- Model: {next(recommender.model.parameters()).device}")
    print(f"- Graph edges: {recommender.graph['edge_index'].device}")
    print(f"- Content features: {recommender.item_content_features.device}")
    
    # Prepare training data
    user_indices = torch.tensor([u for u, _, _ in train_interactions], dtype=torch.long)
    item_indices = torch.tensor([i for _, i, _ in train_interactions], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in train_interactions], dtype=torch.float)
    
    dataset = TensorDataset(user_indices, item_indices, ratings)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Training setup
    optimizer = torch.optim.Adam(
        recommender.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    best_loss = float('inf')
    patience = 0
    
    print("\nStarting training...")
    for epoch in range(config.epochs):
        recommender.model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            user_idx, item_idx, true_ratings = batch
            user_idx = user_idx.to(config.device)
            item_idx = item_idx.to(config.device)
            true_ratings = true_ratings.to(config.device)
            
            # Forward pass with debug checks
            content_features = recommender.item_content_features[item_idx]
            
            pred = recommender.model(
                user_idx, item_idx, content_features,
                recommender.graph['edge_index'],
                recommender.graph['edge_weight']
            )
            
            loss = F.mse_loss(pred, true_ratings)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Epoch processing
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
            torch.save(recommender.model.state_dict(), "best_model.pt")
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    recommender.model.load_state_dict(torch.load("best_model.pt"))
    return recommender