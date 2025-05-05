import torch
import os
import json
from datetime import datetime
from typing import Dict, Optional, List
from config import Config

class SnapshotManager:
    """
    Manages model and graph snapshots for efficient rollback during unlearning.
    """
    def __init__(self, config: 'Config'):
        self.config = config
        os.makedirs(config.snapshot_dir, exist_ok=True)
        self.snapshots = []
        self.load_existing_snapshots()
        
    def load_existing_snapshots(self):
        """Load existing snapshots from disk"""
        if not os.path.exists(self.config.snapshot_dir):
            return
            
        snapshot_files = sorted(
            [f for f in os.listdir(self.config.snapshot_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('_')[1])
        )
        
        for f in snapshot_files[-self.config.max_snapshots:]:  # Only keep most recent
            snapshot_id = int(f.split('_')[1])
            metadata_file = f"snapshot_{snapshot_id}_meta.json"
            
            if os.path.exists(os.path.join(self.config.snapshot_dir, metadata_file)):
                self.snapshots.append({
                    'id': snapshot_id,
                    'file': f,
                    'metadata_file': metadata_file
                })
    
    def create_snapshot(self, 
                       model: torch.nn.Module,
                       graph: Dict[str, torch.Tensor],
                       user_map: Dict,
                       item_map: Dict,
                       interaction_count: int) -> Optional[int]:
        """
        Create a new snapshot of the model and graph state.
        
        Args:
            model: The current model state
            graph: The current graph state
            user_map: User ID to index mapping
            item_map: Item ID to index mapping
            interaction_count: Number of interactions processed
            
        Returns:
            Snapshot ID if created, None if not created due to limits
        """
        # Check if we've reached max snapshots
        if len(self.snapshots) >= self.config.max_snapshots:
            # Remove oldest snapshot
            oldest = self.snapshots.pop(0)
            os.remove(os.path.join(self.config.snapshot_dir, oldest['file']))
            os.remove(os.path.join(self.config.snapshot_dir, oldest['metadata_file']))
        
        # Create new snapshot
        snapshot_id = interaction_count
        snapshot_file = f"snapshot_{snapshot_id}.pt"
        metadata_file = f"snapshot_{snapshot_id}_meta.json"
        
        # Save model and graph state
        torch.save({
            'model_state': model.state_dict(),
            'graph': graph,
            'user_map': user_map,
            'item_map': item_map
        }, os.path.join(self.config.snapshot_dir, snapshot_file))
        
        # Save metadata
        metadata = {
            'interaction_count': interaction_count,
            'created_at': datetime.now().isoformat(),
            'num_users': len(user_map),
            'num_items': len(item_map),
            'num_edges': graph['edge_index'].shape[1]
        }
        
        with open(os.path.join(self.config.snapshot_dir, metadata_file), 'w') as f:
            json.dump(metadata, f)
        
        # Add to tracked snapshots
        self.snapshots.append({
            'id': snapshot_id,
            'file': snapshot_file,
            'metadata_file': metadata_file
        })
        
        return snapshot_id
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get the most recent snapshot"""
        if not self.snapshots:
            return None
            
        latest = self.snapshots[-1]
        return self._load_snapshot(latest['file'], latest['metadata_file'])
    
    def find_best_snapshot_for_unlearning(self,
                                        request_type: str,
                                        user_id: Optional[int] = None,
                                        item_id: Optional[int] = None,
                                        interaction_id: Optional[int] = None) -> Optional[Dict]:
        """
        Find the most appropriate snapshot for an unlearning request.
        
        Args:
            request_type: Type of unlearning request
            user_id: For user deletion
            item_id: For item deletion
            interaction_id: For interaction removal
            
        Returns:
            The best snapshot dictionary or None if not found
        """
        # Simplified implementation - in practice would need to check which snapshots
        # don't contain the entity to be unlearned
        return self.get_latest_snapshot()
    
    def _load_snapshot(self, snapshot_file: str, metadata_file: str) -> Dict:
        """Load a snapshot from disk"""
        data = torch.load(os.path.join(self.config.snapshot_dir, snapshot_file))
        
        with open(os.path.join(self.config.snapshot_dir, metadata_file), 'r') as f:
            metadata = json.load(f)
        
        return {**data, **metadata}
    
    def get_interactions_since(self, interaction_count: int) -> List:
        """
        Get interactions that occurred after a certain snapshot.
        In a real implementation, this would query your interaction log.
        """
        # Placeholder - would need proper interaction tracking
        return []