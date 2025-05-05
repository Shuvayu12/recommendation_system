import torch
from typing import Dict, Optional
from datetime import datetime
import os

from config import Config
from models.hybrid import HybridRecommender

class UnlearningManager:
    """
    Manages the unlearning process with snapshot and graph rewiring strategies.
    """
    def __init__(self, config: 'Config'):
        self.config = config
        self.unlearning_requests = []
        self.affected_nodes_tracker = {}  # Track affected nodes for each request
        
    def register_request(self, 
                        request_type: str,
                        user_id: Optional[int] = None,
                        item_id: Optional[int] = None,
                        interaction_id: Optional[int] = None):
        """
        Register an unlearning request.
        
        Args:
            request_type: Type of request ('user', 'item', or 'interaction')
            user_id: For user deletion requests
            item_id: For item deletion requests
            interaction_id: For specific interaction removal
        """
        request = {
            'type': request_type,
            'user_id': user_id,
            'item_id': item_id,
            'interaction_id': interaction_id,
            'timestamp': datetime.now(),
            'processed': False
        }
        self.unlearning_requests.append(request)
        
    def process_pending_requests(self, recommender: 'HybridRecommender') -> int:
        """
        Process all pending unlearning requests.
        
        Args:
            recommender: The recommender system instance
            
        Returns:
            Number of requests processed
        """
        processed = 0
        for request in self.unlearning_requests:
            if not request['processed']:
                success = recommender.process_unlearning_request(
                    request['type'],
                    request['user_id'],
                    request['item_id'],
                    request['interaction_id']
                )
                
                if success:
                    request['processed'] = True
                    processed += 1
                    self._track_affected_nodes(request, recommender)
        
        return processed
    
    def _track_affected_nodes(self, 
                            request: Dict,
                            recommender: 'HybridRecommender'):
        """
        Track which nodes were affected by an unlearning request.
        """
        request_id = len(self.unlearning_requests) - 1
        
        if request['type'] == 'user':
            user_idx = recommender.user_map.get(request['user_id'])
            if user_idx is not None:
                self.affected_nodes_tracker[request_id] = {
                    'users': torch.tensor([user_idx], device=self.config.device),
                    'items': torch.tensor([], device=self.config.device)
                }
                
        elif request['type'] == 'item':
            item_idx = recommender.item_map.get(request['item_id'])
            if item_idx is not None:
                self.affected_nodes_tracker[request_id] = {
                    'users': torch.tensor([], device=self.config.device),
                    'items': torch.tensor([item_idx], device=self.config.device)
                }
                
        elif request['type'] == 'interaction':
            # This would need proper interaction tracking to implement
            pass
    
    def get_affected_nodes_for_request(self, request_id: int) -> Optional[Dict]:
        """
        Get the nodes affected by a specific unlearning request.
        """
        return self.affected_nodes_tracker.get(request_id)