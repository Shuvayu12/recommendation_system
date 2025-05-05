from unittest.mock import MagicMock
import pytest
from models.unlearning import UnlearningManager
from config import config

def test_unlearning_request_registration():
    """Test unlearning request registration"""
    manager = UnlearningManager(config)
    
    # Register requests
    manager.register_request('user', user_id=1)
    manager.register_request('item', item_id=101)
    manager.register_request('interaction', interaction_id=1001)
    
    assert len(manager.unlearning_requests) == 3
    assert not any(req['processed'] for req in manager.unlearning_requests)

def test_request_processing():
    """Test processing of unlearning requests"""
    manager = UnlearningManager(config)
    recommender_mock = MagicMock()
    
    # Setup mock to return True for process_unlearning_request
    recommender_mock.process_unlearning_request.return_value = True
    
    # Register and process requests
    manager.register_request('user', user_id=1)
    manager.register_request('item', item_id=101)
    
    processed = manager.process_pending_requests(recommender_mock)
    
    assert processed == 2
    assert all(req['processed'] for req in manager.unlearning_requests[:2])
    assert recommender_mock.process_unlearning_request.call_count == 2

def test_affected_node_tracking():
    """Test tracking of affected nodes during unlearning"""
    manager = UnlearningManager(config)
    recommender_mock = MagicMock()
    
    # Setup mock recommender with user and item maps
    recommender_mock.user_map = {1: 0, 2: 1}
    recommender_mock.item_map = {101: 0, 102: 1}
    
    # Register and process user unlearning
    manager.register_request('user', user_id=1)
    manager.process_pending_requests(recommender_mock)
    
    affected = manager.get_affected_nodes_for_request(0)
    assert affected is not None
    assert len(affected['users']) == 1
    assert len(affected['items']) == 0