import torch
from typing import List, Tuple, Dict

import torch
from typing import List, Tuple, Dict

def build_graph_from_interactions(interactions: List[Tuple[int, int, float]],
                                num_users: int,
                                num_items: int,
                                device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Builds a bipartite graph with proper device placement and validation"""
    if not interactions:
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long, device=device),
            'edge_weight': torch.empty((0,), dtype=torch.float, device=device),
            'num_nodes': num_users + num_items
        }
    
    # Validate input ranges
    user_indices = torch.tensor([u for u, _, _ in interactions], dtype=torch.long)
    item_indices = torch.tensor([i for _, i, _ in interactions], dtype=torch.long)
    
    assert user_indices.max() < num_users, "User index out of bounds"
    assert item_indices.max() < num_items, "Item index out of bounds"
    
    # Build symmetric edges
    edge_index = torch.stack([
        torch.cat([user_indices, item_indices + num_users]),
        torch.cat([item_indices + num_users, user_indices])
    ], dim=0).to(device)
    
    # Normalize ratings to [0,1] range
    ratings = torch.tensor([r for _, _, r in interactions], dtype=torch.float)
    edge_weight = torch.cat([ratings, ratings]).to(device) / 5.0  # Assuming 1-5 rating scale
    
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': num_users + num_items
    }