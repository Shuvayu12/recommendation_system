import torch
from typing import List, Tuple

def build_graph_from_interactions(interactions: List[Tuple[int, int, float]],
                                num_users: int,
                                num_items: int,
                                device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Build a bipartite graph from user-item interactions.
    
    Args:
        interactions: List of (user_idx, item_idx, rating) tuples
        num_users: Number of users
        num_items: Number of items
        device: Device to store tensors on
        
    Returns:
        Dictionary with 'edge_index' and 'edge_weight' tensors
    """
    if not interactions:
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
            'edge_weight': torch.empty((0,), dtype=torch.float, device=device)
        }
    
    # Convert to tensors
    user_indices = torch.tensor([u for u, _, _ in interactions], dtype=torch.long)
    item_indices = torch.tensor([i for _, i, _ in interactions], dtype=torch.long)
    ratings = torch.tensor([r for _, _, r in interactions], dtype=torch.float)
    
    # Create edge index (items are offset by num_users in the graph)
    edge_index = torch.stack([
        torch.cat([user_indices, item_indices + num_users]),
        torch.cat([item_indices + num_users, user_indices])
    ], dim=0)
    
    # Edge weights (ratings are duplicated for both directions)
    edge_weight = torch.cat([ratings, ratings])
    
    return {
        'edge_index': edge_index.to(device),
        'edge_weight': edge_weight.to(device)
    }