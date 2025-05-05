import pytest
import os
from utils.data_loader import load_movielens_data, _process_item_content

@pytest.fixture
def mock_movielens_data(tmp_path):
    """Create mock MovieLens data files for testing"""
    # Create ratings file
    ratings_content = """1::101::5::964982703
1::102::3::964981247
2::101::4::964982224
2::103::5::964981343
3::102::4::964982453
3::103::3::964982608"""
    
    # Create movies file
    movies_content = """101::Toy Story (1995)::Animation|Children's|Comedy
102::Jumanji (1995)::Adventure|Children's|Fantasy
103::Grumpier Old Men (1995)::Comedy|Romance"""
    
    # Write files
    data_dir = tmp_path / "ml-1m"
    data_dir.mkdir()
    
    (data_dir / "ratings.dat").write_text(ratings_content)
    (data_dir / "movies.dat").write_text(movies_content)
    
    return str(data_dir)

def test_data_loading(mock_movielens_data):
    """Test loading of MovieLens data"""
    data = load_movielens_data(mock_movielens_data, min_rating=4)
    
    assert len(data['user_ids']) == 3
    assert len(data['item_ids']) == 3
    assert len(data['train_interactions']) + len(data['test_interactions']) == 4  # 2 ratings filtered out
    assert 101 in data['item_features']
    assert data['item_features'][101].shape[0] > 0  # Should have some features

def test_item_feature_processing():
    """Test processing of item content features"""
    mock_movies = {
        'item_id': [101, 102],
        'title': ['Toy Story (1995)', 'Jumanji (1995)'],
        'genres': ['Animation|Children\'s|Comedy', 'Adventure|Children\'s|Fantasy']
    }
    
    # Convert to DataFrame
    import pandas as pd
    movies_df = pd.DataFrame(mock_movies)
    
    features = _process_item_content(movies_df)
    
    assert len(features) == 2
    assert 101 in features
    assert features[101].dtype == torch.float32