import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from models.hybrid import HybridRecommender
from models.gnn import HybridGNN
from config import config

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return {
        'user_ids': [1, 2, 3],
        'item_ids': [101, 102, 103, 104],
        'item_features': {
            101: torch.rand(50),
            102: torch.rand(50),
            103: torch.rand(50),
            104: torch.rand(50)
        },
        'train_interactions': [
            (1, 101, 4.0),
            (1, 102, 3.5),
            (2, 103, 5.0),
            (3, 104, 4.5)
        ]
    }

def test_hybrid_recommender_initialization(sample_data):
    """Test recommender initialization"""
    recommender = HybridRecommender(config)
    recommender.initialize(
        sample_data['user_ids'],
        sample_data['item_ids'],
        sample_data['item_features']
    )
    
    assert len(recommender.user_map) == 3
    assert len(recommender.item_map) == 4
    assert recommender.model is not None
    assert recommender.graph is not None

def test_graph_update(sample_data):
    """Test graph update functionality"""
    recommender = HybridRecommender(config)
    recommender.initialize(
        sample_data['user_ids'],
        sample_data['item_ids'],
        sample_data['item_features']
    )
    
    initial_edge_count = recommender.graph['edge_index'].shape[1]
    recommender.update_graph(sample_data['train_interactions'])
    
    # Should have 4 interactions * 2 (bidirectional)
    assert recommender.graph['edge_index'].shape[1] == initial_edge_count + 8

def test_recommendation_generation(sample_data):
    """Test recommendation generation"""
    recommender = HybridRecommender(config)
    recommender.initialize(
        sample_data['user_ids'],
        sample_data['item_ids'],
        sample_data['item_features']
    )
    recommender.update_graph(sample_data['train_interactions'])
    
    # Mock model prediction
    with patch.object(recommender.model, 'forward') as mock_forward:
        mock_forward.return_value = torch.tensor([4.5, 3.0, 2.5, 1.0])
        recommendations = recommender.recommend(1, k=2)
        
        assert len(recommendations) == 2
        assert recommendations[0][1] >= recommendations[1][1]  # Check sorting

def test_unlearning_rollback(sample_data):
    """Test unlearning via snapshot rollback"""
    recommender = HybridRecommender(config)
    recommender.initialize(
        sample_data['user_ids'],
        sample_data['item_ids'],
        sample_data['item_features']
    )
    
    # Create initial state
    initial_interaction_count = recommender.interaction_count
    recommender.update_graph(sample_data['train_interactions'])
    
    # Mock snapshot manager
    snapshot_mock = MagicMock()
    snapshot_mock.get_latest_snapshot.return_value = {
        'model_state': recommender.model.state_dict(),
        'graph': recommender.graph,
        'user_map': recommender.user_map,
        'item_map': recommender.item_map,
        'interaction_count': initial_interaction_count
    }
    snapshot_mock.get_interactions_since.return_value = sample_data['train_interactions']
    recommender.snapshot_manager = snapshot_mock
    
    # Test unlearning
    success = recommender.process_unlearning_request('user', user_id=1)
    assert success
    assert recommender.interaction_count == len(sample_data['train_interactions'])