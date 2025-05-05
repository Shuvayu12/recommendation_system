import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np
from datetime import datetime

from models.hybrid import HybridRecommender
from utils.data_loader import load_movielens_data
from utils.metrics import calculate_metrics
from config import config

def train_recommender():
    """Main training function for the hybrid recommender system."""
    # Load and preprocess data
    print("Loading MovieLens data...")
    data = load_movielens_data(config.data_path, config.min_rating)
    
    # Initialize recommender
    print("Initializing recommender system...")
    recommender = HybridRecommender(config)
    recommender.initialize(
        user_ids=data['user_ids'],
        item_ids=data['item_ids'],
        item_content_features=data['item_features']
    )
    
    # Prepare training data
    train_interactions = data['train_interactions']
    test_interactions = data['test_interactions']
    
    # Convert to internal indices
    train_interactions_internal = [
        (recommender.user_map[u], recommender.item_map[i], r)
        for u, i, r in train_interactions
    ]
    
    # Initial graph update
    print("Building initial graph...")
    recommender.update_graph(train_interactions)
    
    # Training setup
    optimizer = torch.optim.Adam(
        recommender.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create DataLoader
    user_indices = torch.tensor([u for u, _, _ in train_interactions_internal])
    item_indices = torch.tensor([i for _, i, _ in train_interactions_internal])
    ratings = torch.tensor([r for _, _, r in train_interactions_internal])
    
    dataset = TensorDataset(user_indices, item_indices, ratings)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        recommender.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            user_idx, item_idx, true_ratings = batch
            user_idx = user_idx.to(config.device)
            item_idx = item_idx.to(config.device)
            true_ratings = true_ratings.to(config.device)
            
            # Get content features for batch items
            content_features = recommender.item_content_features[item_idx]
            
            # Forward pass
            pred_ratings = recommender.model(
                user_idx, item_idx, content_features,
                recommender.graph['edge_index'],
                recommender.graph['edge_weight']
            )
            
            # Calculate loss
            loss = F.mse_loss(pred_ratings, true_ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average epoch loss
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss:.4f}")
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluation
    print("Evaluating on test set...")
    test_results = evaluate_recommender(recommender, test_interactions)
    print("\nTest Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    return recommender

def evaluate_recommender(recommender: HybridRecommender,
                       test_interactions: List[Tuple[int, int, float]],
                       k: int = 10) -> Dict[str, float]:
    """
    Evaluate the recommender system on test interactions.
    
    Args:
        recommender: The recommender system instance
        test_interactions: List of (user_id, item_id, rating) tuples
        k: Number of recommendations to consider for metrics
        
    Returns:
        Dictionary of metric names to values
    """
    # Convert to internal indices and filter unknown users/items
    test_data = []
    for u, i, r in test_interactions:
        if u in recommender.user_map and i in recommender.item_map:
            test_data.append((
                recommender.user_map[u],
                recommender.item_map[i],
                r
            ))
    
    if not test_data:
        raise ValueError("No valid test interactions after filtering")
    
    # Generate recommendations and calculate metrics
    all_user_recs = {}
    for user_idx, _, _ in test_data:
        user_id = recommender.reverse_user_map[user_idx]
        try:
            recs = recommender.recommend(user_id, k=k)
            all_user_recs[user_idx] = [recommender.item_map[i] for i, _ in recs]
        except ValueError:
            continue
    
    # Calculate metrics
    return calculate_metrics(test_data, all_user_recs, k)

if __name__ == "__main__":
    recommender = train_recommender()
    
    # Save the trained recommender
    torch.save(recommender, "trained_recommender.pt")
    print("Training complete. Model saved to 'trained_recommender.pt'")