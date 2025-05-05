from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

def calculate_metrics(test_interactions: List[Tuple[int, int, float]],
                     all_user_recs: Dict[int, List[int]],
                     k: int = 10) -> Dict[str, float]:
    """
    Calculate recommendation metrics (Precision@k, Recall@k, NDCG@k).
    
    Args:
        test_interactions: List of (user_idx, item_idx, rating) tuples
        all_user_recs: Dict mapping user_idx to list of recommended item indices
        k: Number of recommendations to consider
        
    Returns:
        Dictionary of metric names to values
    """
    # Organize test interactions by user
    test_user_items = defaultdict(list)
    for user_idx, item_idx, rating in test_interactions:
        test_user_items[user_idx].append(item_idx)
    
    # Initialize metrics
    precisions = []
    recalls = []
    ndcgs = []
    
    for user_idx, rec_items in all_user_recs.items():
        if user_idx not in test_user_items or not rec_items:
            continue
            
        test_items = test_user_items[user_idx]
        if not test_items:
            continue
            
        # Precision and Recall
        hits = len(set(rec_items[:k]) & set(test_items))
        precisions.append(hits / k)
        recalls.append(hits / min(len(test_items), k))
        
        # NDCG
        relevance = np.zeros(k)
        for i, item in enumerate(rec_items[:k]):
            if item in test_items:
                relevance[i] = 1  # Binary relevance
                
        dcg = _calculate_dcg(relevance)
        idcg = _calculate_dcg(np.sort(relevance)[::-1])
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    # Aggregate metrics
    return {
        'precision@k': np.mean(precisions) if precisions else 0,
        'recall@k': np.mean(recalls) if recalls else 0,
        'ndcg@k': np.mean(ndcgs) if ndcgs else 0,
        'coverage': len(all_user_recs) / len(test_user_items) if test_user_items else 0
    }

def _calculate_dcg(relevance: np.ndarray) -> float:
    """Calculate Discounted Cumulative Gain"""
    return sum((2 ** rel - 1) / np.log2(idx + 2) 
               for idx, rel in enumerate(relevance))