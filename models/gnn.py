import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from torch_geometric.nn import GATConv
from typing import Dict, Tuple, Optional

class HybridGNN(nn.Module):
    def __init__(self, num_users: int, num_items: int, content_feat_dim: int, config: 'Config'):
        super().__init__()
        self.config = config
        
        # User and item embeddings
        self.user_emb = nn.Embedding(num_users, config.embedding_dim)
        self.item_emb = nn.Embedding(num_items, config.embedding_dim)
        
        # Content feature projection
        self.content_proj = nn.Linear(content_feat_dim, config.embedding_dim)
        
        # GNN layers - let PyG handle initialization
        self.conv1 = GATConv(
            in_channels=config.embedding_dim * 2,
            out_channels=config.embedding_dim,
            heads=2,
            dropout=config.dropout,
            add_self_loops=False,
            bias=True  # Explicitly enable bias
        )
        self.conv2 = GATConv(
            in_channels=config.embedding_dim * 2,  # heads=2 multiplies dims
            out_channels=config.embedding_dim,
            heads=1,
            dropout=config.dropout,
            add_self_loops=False,
            bias=True
        )
        
        # Prediction head
        self.predict = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, 1)
        )
        
        # Initialize only components WE control
        self._init_weights()
        self.num_users = num_users
        self.num_items = num_items
    
    def _init_weights(self):
        """Initialize only embeddings and prediction head"""
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.content_proj.weight)
        
        for layer in self.predict:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, item_ids, content_features, edge_index, edge_weight=None):
        # Device consistency
        assert user_ids.device == item_ids.device == content_features.device == edge_index.device
        if edge_weight is not None:
            assert edge_weight.device == edge_index.device
        
        # Get embeddings
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        c_emb = self.content_proj(content_features)
        
        # Combine item features
        item_emb = torch.cat([i_emb, c_emb], dim=1)
        
        # Create full graph embeddings
        x = torch.zeros((self.num_users + self.num_items, 2 * self.config.embedding_dim),
                      device=self.config.device)
        x[user_ids] = torch.cat([u_emb, torch.zeros_like(c_emb)], dim=1)
        x[self.num_users + item_ids] = item_emb
        
        # Add self-loops
        from torch_geometric.utils import add_self_loops
        edge_index, edge_weight = add_self_loops(
            edge_index, 
            edge_weight,
            num_nodes=self.num_users + self.num_items,
            fill_value=1.0
        )
        
        # GNN processing
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
        # Get final embeddings
        user_emb_gnn = x[user_ids]
        item_emb_gnn = x[self.num_users + item_ids]
        
        # Prediction
        combined = torch.cat([u_emb, item_emb_gnn, user_emb_gnn], dim=1)
        return self.predict(combined).squeeze()
    
    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        return {
            'user_emb': self.user_emb.weight.detach(),
            'item_emb': self.item_emb.weight.detach()
        }