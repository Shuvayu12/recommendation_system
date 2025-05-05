import pytest
import torch
from models.gnn import HybridGNN

def test_hybrid_gnn_initialization():
    """Test GNN model initialization"""
    model = HybridGNN(
        num_users=10,
        num_items=20,
        content_feat_dim=50,
        config=config
    )
    
    assert model.user_emb.num_embeddings == 10
    assert model.item_emb.num_embeddings == 20
    assert model.content_proj.in_features == 50
    assert model.conv1.out_channels == config.embedding_dim

def test_hybrid_gnn_forward():
    """Test forward pass of GNN model"""
    model = HybridGNN(
        num_users=10,
        num_items=20,
        content_feat_dim=50,
        config=config
    ).to(config.device)
    
    # Test data
    user_ids = torch.tensor([0, 1], device=config.device)
    item_ids = torch.tensor([0, 1], device=config.device)
    content_features = torch.rand(2, 50, device=config.device)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=config.device)
    edge_weight = torch.tensor([1.0, 1.0], device=config.device)
    
    # Forward pass
    output = model(user_ids, item_ids, content_features, edge_index, edge_weight)
    
    assert output.shape == (2,)
    assert not torch.isnan(output).any()

def test_embedding_retrieval():
    """Test embedding retrieval method"""
    model = HybridGNN(
        num_users=10,
        num_items=20,
        content_feat_dim=50,
        config=config
    )
    
    embeddings = model.get_embeddings()
    assert 'user_emb' in embeddings
    assert 'item_emb' in embeddings
    assert embeddings['user_emb'].shape == (10, config.embedding_dim)
    assert embeddings['item_emb'].shape == (20, config.embedding_dim)