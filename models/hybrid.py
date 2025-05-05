import torch
from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
from datetime import datetime
import os

from .gnn import HybridGNN
from utils.graph_ops import build_graph_from_interactions
from utils.snapshot_manager import SnapshotManager

class HybridRecommender:
    """
    Main recommendation system class combining CF and CBF with GNN,
    with unlearning capabilities.
    """
    def __init__(self, config: 'Config'):
        self.config = config
        self.model = None
        self.graph = None
        self.user_map = {}  # user_id -> index
        self.item_map = {}  # item_id -> index
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.interaction_count = 0
        self.snapshot_manager = SnapshotManager(config)
        
        # Track recent interactions for rewiring
        self.recent_interactions = defaultdict(list)
        self.current_window = 0
        
    def initialize(self, 
                  user_ids: List[int], 
                  item_ids: List[int],
                  item_content_features: Dict[int, torch.Tensor]):
        """
        Initialize the recommender system with users, items and content features.
        
        Args:
            user_ids: List of unique user IDs
            item_ids: List of unique item IDs
            item_content_features: Dict mapping item_id to content feature tensor
        """
        # Create mapping from original IDs to internal indices
        self.user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_map.items()}
        self.item_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}
        
        # Store content features
        self.item_content_features = torch.stack([
            item_content_features[iid] for iid in item_ids
        ]).to(self.config.device)
        
        # Initialize model
        self.model = HybridGNN(
            num_users=len(user_ids),
            num_items=len(item_ids),
            content_feat_dim=self.item_content_features.shape[1],
            config=self.config
        ).to(self.config.device)
        
        # Initialize empty graph
        self.graph = {
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_weight': torch.empty((0,), dtype=torch.float)
        }
        
    def update_graph(self, interactions: List[Tuple[int, int, float]]):
        """
        Update the user-item interaction graph with new interactions.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
        """
        # Convert to internal indices
        internal_interactions = [
            (self.user_map[u], self.item_map[i], r)
            for u, i, r in interactions
            if u in self.user_map and i in self.item_map
        ]
        
        # Update recent interactions for rewiring
        for u, i, r in internal_interactions:
            self.recent_interactions[self.current_window].append((u, i, r))
        
        # Build new graph
        self.graph = build_graph_from_interactions(
            internal_interactions,
            num_users=len(self.user_map),
            num_items=len(self.item_map),
            device=self.config.device
        )
        
        self.interaction_count += len(internal_interactions)
        
        # Create snapshot if needed
        if self.interaction_count % self.config.snapshot_interval == 0:
            self.snapshot_manager.create_snapshot(
                model=self.model,
                graph=self.graph,
                user_map=self.user_map,
                item_map=self.item_map,
                interaction_count=self.interaction_count
            )
            
        # Update time window
        if len(self.recent_interactions[self.current_window]) >= self.config.rewiring_window:
            self.current_window += 1
    
    def recommend(self, 
                 user_id: int, 
                 k: int = 10,
                 exclude_interacted: bool = True) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: Original user ID
            k: Number of recommendations to return
            exclude_interacted: Whether to exclude already interacted items
            
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_map:
            raise ValueError(f"Unknown user ID: {user_id}")
            
        user_idx = self.user_map[user_id]
        user_tensor = torch.tensor([user_idx], device=self.config.device)
        
        # Get all item indices
        item_indices = torch.arange(len(self.item_map), device=self.config.device)
        item_indices = item_indices.repeat(len(user_tensor))
        user_indices = user_tensor.repeat_interleave(len(self.item_map))
        
        # Get content features for all items
        content_features = self.item_content_features[item_indices]
        
        # Predict scores
        with torch.no_grad():
            scores = self.model(
                user_indices,
                item_indices,
                content_features,
                self.graph['edge_index'],
                self.graph['edge_weight']
            )
        
        # Convert to numpy and process
        scores = scores.cpu().numpy()
        item_scores = list(zip(item_indices.cpu().numpy(), scores))
        
        # Sort by score descending
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert back to original IDs
        recommendations = [
            (self.reverse_item_map[i], s)
            for i, s in item_scores[:k]
        ]
        
        return recommendations
    
    def process_unlearning_request(self, 
                                  request_type: str,
                                  user_id: Optional[int] = None,
                                  item_id: Optional[int] = None,
                                  interaction_id: Optional[int] = None):
        """
        Process an unlearning request.
        
        Args:
            request_type: Type of request ('user', 'item', or 'interaction')
            user_id: For user deletion requests
            item_id: For item deletion requests
            interaction_id: For specific interaction removal
            
        Returns:
            bool: Whether unlearning was successful
        """
        # Check if we can use snapshot rollback
        if self._can_use_snapshot_rollback():
            return self._unlearn_via_snapshot_rollback(
                request_type, user_id, item_id, interaction_id)
        else:
            return self._unlearn_via_graph_rewiring(
                request_type, user_id, item_id, interaction_id)
    
    def _can_use_snapshot_rollback(self) -> bool:
        """Check if snapshot rollback is feasible"""
        latest_snapshot = self.snapshot_manager.get_latest_snapshot()
        if not latest_snapshot:
            return False
            
        steps_behind = self.interaction_count - latest_snapshot['interaction_count']
        return steps_behind <= self.config.max_rollback_steps * self.config.snapshot_interval
    
    def _unlearn_via_snapshot_rollback(self, 
                                      request_type: str,
                                      user_id: Optional[int],
                                      item_id: Optional[int],
                                      interaction_id: Optional[int]) -> bool:
        """Unlearn by rolling back to a snapshot and replaying interactions"""
        # Find the appropriate snapshot
        target_snapshot = self.snapshot_manager.find_best_snapshot_for_unlearning(
            request_type, user_id, item_id, interaction_id)
        
        if not target_snapshot:
            return False
            
        # Restore from snapshot
        self.model.load_state_dict(target_snapshot['model_state'])
        self.graph = target_snapshot['graph']
        self.user_map = target_snapshot['user_map']
        self.item_map = target_snapshot['item_map']
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        # Replay interactions since snapshot
        interactions_to_replay = self.snapshot_manager.get_interactions_since(
            target_snapshot['interaction_count'])
        self.update_graph(interactions_to_replay)
        
        return True
    
    def _unlearn_via_graph_rewiring(self,
                                  request_type: str,
                                  user_id: Optional[int],
                                  item_id: Optional[int],
                                  interaction_id: Optional[int]) -> bool:
        """Unlearn by modifying the graph structure and performing gradient surgery"""
        # Remove the requested data from the graph
        modified = False
        
        if request_type == 'user' and user_id is not None:
            modified = self._remove_user_from_graph(user_id)
        elif request_type == 'item' and item_id is not None:
            modified = self._remove_item_from_graph(item_id)
        elif request_type == 'interaction' and interaction_id is not None:
            modified = self._remove_interaction_from_graph(interaction_id)
        
        if not modified:
            return False
            
        # Relearn with gradient surgery
        self._relearn_with_gradient_surgery()
        
        return True
    
    def _remove_user_from_graph(self, user_id: int) -> bool:
        """Remove a user and all their interactions from the graph"""
        if user_id not in self.user_map:
            return False
            
        user_idx = self.user_map[user_id]
        
        # Remove edges connected to this user
        mask = self.graph['edge_index'][0] != user_idx
        self.graph['edge_index'] = self.graph['edge_index'][:, mask]
        self.graph['edge_weight'] = self.graph['edge_weight'][mask]
        
        # Remove from mappings
        del self.user_map[user_id]
        del self.reverse_user_map[user_idx]
        
        return True
    
    def _remove_item_from_graph(self, item_id: int) -> bool:
        """Remove an item and all its interactions from the graph"""
        if item_id not in self.item_map:
            return False
            
        item_idx = self.item_map[item_id]
        num_users = len(self.user_map)
        
        # Remove edges connected to this item (items are in second half of graph)
        mask = self.graph['edge_index'][1] != (item_idx + num_users)
        self.graph['edge_index'] = self.graph['edge_index'][:, mask]
        self.graph['edge_weight'] = self.graph['edge_weight'][mask]
        
        # Remove from mappings and content features
        del self.item_map[item_id]
        del self.reverse_item_map[item_idx]
        self.item_content_features = torch.cat([
            self.item_content_features[:item_idx],
            self.item_content_features[item_idx+1:]
        ])
        
        return True
    
    def _remove_interaction_from_graph(self, interaction_id: int) -> bool:
        """Remove a specific interaction from the graph"""
        # In a real implementation, we'd need to track interaction IDs
        # For simplicity, we'll assume we can identify the interaction
        # This would need to be implemented based on your tracking system
        return False
    
    def _relearn_with_gradient_surgery(self):
        """Relearn the model with gradient surgery on affected subgraph"""
        # Identify affected nodes (simplified - in practice would need proper tracking)
        affected_nodes = self._identify_affected_nodes()
        
        # Create data loader for affected subgraph
        train_loader = self._create_subgraph_loader(affected_nodes)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop with gradient surgery
        for epoch in range(self.config.gradient_surgery_epochs):
            self.model.train()
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                user_idx, item_idx, ratings = batch
                content_features = self.item_content_features[item_idx]
                pred = self.model(
                    user_idx, item_idx, content_features,
                    self.graph['edge_index'],
                    self.graph['edge_weight']
                )
                
                # Loss calculation
                loss = F.mse_loss(pred, ratings)
                
                # Backward pass with gradient surgery
                loss.backward()
                
                # Gradient surgery: zero out gradients for unaffected parameters
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if 'user_emb' in name or 'item_emb' in name:
                            # Only update embeddings for affected nodes
                            mask = torch.zeros_like(param.grad)
                            if 'user_emb' in name:
                                mask[affected_nodes['users']] = 1
                            else:
                                mask[affected_nodes['items']] = 1
                            param.grad *= mask
                
                optimizer.step()
    
    def _identify_affected_nodes(self) -> Dict[str, torch.Tensor]:
        """
        Identify nodes affected by the unlearning operation.
        Simplified implementation - in practice would need proper tracking.
        """
        # This is a placeholder - real implementation would track propagation
        return {
            'users': torch.arange(len(self.user_map), device=self.config.device),
            'items': torch.arange(len(self.item_map), device=self.config.device)
        }
    
    def _create_subgraph_loader(self, affected_nodes: Dict[str, torch.Tensor]):
        """
        Create a data loader for the affected subgraph.
        Simplified implementation.
        """
        # In practice, would sample from recent interactions involving affected nodes
        # For simplicity, we'll use all current edges
        edge_index = self.graph['edge_index']
        edge_weight = self.graph['edge_weight']
        
        # Convert to user-item pairs and ratings
        num_users = len(self.user_map)
        user_indices = edge_index[0]
        item_indices = edge_index[1] - num_users  # Convert back to item indices
        
        dataset = torch.utils.data.TensorDataset(
            user_indices, item_indices, edge_weight)
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )