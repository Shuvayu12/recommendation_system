import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, Tuple

class HybridGNN(nn.Module):
    """
    Hybrid GNN model combining collaborative and content-based features.
    Uses Graph Attention Networks (GAT) for better interpretability.
    """
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 content_feat_dim: int,
                 config: 'Config'):
        super().__init__()
        self.config = config
        
        # User and item embeddings (collaborative)
        self.user_emb = nn.Embedding(num_users, config.embedding_dim)
        self.item_emb = nn.Embedding(num_items, config.embedding_dim)
        
        # Content feature projection
        self.content_proj = nn.Linear(content_feat_dim, config.embedding_dim)
        
        # GNN layers
        self.conv1 = GATConv(config.embedding_dim * 2, config.embedding_dim, heads=2)
        self.conv2 = GATConv(config.embedding_dim * 2, config.embedding_dim, heads=1)
        
        # Prediction layers
        self.predict = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.content_proj.weight)
        
    def forward(self, 
                user_ids: torch.Tensor, 
                item_ids: torch.Tensor, 
                content_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the hybrid GNN.
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            content_features: Tensor of item content features
            edge_index: Graph connectivity in COO format
            edge_weight: Optional edge weights
            
        Returns:
            Predicted scores for user-item pairs
        """
        # Get collaborative embeddings
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        
        # Get content-based embeddings
        c_emb = self.content_proj(content_features)
        
        # Combine item embeddings
        item_emb = torch.cat([i_emb, c_emb], dim=1)
        
        # Create full graph embeddings
        x = torch.cat([
            torch.cat([u_emb, torch.zeros_like(c_emb)], dim=1),
            item_emb
        ])
        
        # GNN processing
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
        # Separate user and item embeddings after GNN
        user_emb_gnn = x[user_ids]
        item_emb_gnn = x[len(u_emb) + item_ids]
        
        # Final prediction
        combined = torch.cat([u_emb, item_emb_gnn, user_emb_gnn], dim=1)
        return self.predict(combined).squeeze()
    
    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get current user and item embeddings"""
        return {
            'user_emb': self.user_emb.weight.detach(),
            'item_emb': self.item_emb.weight.detach()
        }