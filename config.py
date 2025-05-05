import os
from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class Config:
    # Data configuration
    data_path: str = "data/movielens/"
    min_rating: float = 3.5  # Minimum rating to consider as positive interaction
    test_size: float = 0.2   # Test set size
    
    # Model configuration
    embedding_dim: int = 64
    gnn_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Training configuration
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 5
    
    # Snapshot configuration
    snapshot_dir: str = "data/snapshots/"
    max_snapshots: int = 10  # Maximum number of snapshots to keep
    snapshot_interval: int = 10000  # Create snapshot every N interactions
    
    # Unlearning configuration
    max_rollback_steps: int = 3  # Maximum snapshots to roll back
    rewiring_window: int = 1000  # Time window for graph rewiring (in interactions)
    gradient_surgery_epochs: int = 3  # Epochs for relearning after unlearning
    
    # Performance configuration
    num_workers: int = os.cpu_count() or 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Global configuration instance
config = Config()