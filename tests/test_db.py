import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, StaticPool

from tests.conftest import MOCK_METADATA


@pytest.fixture()
def db_client():
    engine = create_engine(
        'sqlite://',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    with (
        patch('api.main.load_model', return_value=(MagicMock(), MagicMock(), MagicMock(), 'cpu')),
        patch('api.main.load_index', return_value=(MagicMock(), MOCK_METADATA)),
        patch('api.main.load_curation_data', return_value=(np.array([0, 1]), np.array([0.1, 0.9]))),
        patch('api.main.get_cluster_choices', return_value=[('Cluster 0 — person (2 images)', 0)]),
        patch('api.main.create_db_and_tables'),
    ):
        from api.main import app
        from api.database import get_session

        SQLModel.metadata.create_all(engine)

        def override_get_session():
            with Session(engine) as session:
                yield session

        app.dependency_overrides[get_session] = override_get_session

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()


def test_history_empty(db_client):
    response = db_client.get('/history')
    assert response.status_code == 200
    assert response.json() == {'history': []}


def test_search_logs_history(db_client):
    mock_results = MOCK_METADATA.copy()
    mock_results['score'] = [0.9, 0.8]
    with patch('api.main.encode_text', return_value=np.zeros((1, 512))), \
         patch('api.main.search', return_value=mock_results):
        db_client.post('/search', json={'query': 'a cat', 'k': 2})
    history = db_client.get('/history').json()['history']
    assert len(history) == 1
    assert history[0]['query'] == 'a cat'
    assert history[0]['result_count'] == 2


def test_history_newest_first(db_client):
    mock_results = MOCK_METADATA.copy()
    mock_results['score'] = [0.9, 0.8]
    with patch('api.main.encode_text', return_value=np.zeros((1, 512))), \
         patch('api.main.search', return_value=mock_results):
        db_client.post('/search', json={'query': 'first', 'k': 2})
        db_client.post('/search', json={'query': 'second', 'k': 2})
    history = db_client.get('/history').json()['history']
    assert history[0]['query'] == 'second'
    assert history[1]['query'] == 'first'


def test_add_favorite(db_client):
    response = db_client.post('/favorites/000001.jpg')
    assert response.status_code == 200
    assert response.json()['file_name'] == '000001.jpg'


def test_add_favorite_idempotent(db_client):
    db_client.post('/favorites/000001.jpg')
    response = db_client.post('/favorites/000001.jpg')
    assert response.status_code == 200
    favorites = db_client.get('/favorites').json()['favorites']
    assert len(favorites) == 1


def test_list_favorites(db_client):
    db_client.post('/favorites/000001.jpg')
    db_client.post('/favorites/000002.jpg')
    names = [f['file_name'] for f in db_client.get('/favorites').json()['favorites']]
    assert '000001.jpg' in names
    assert '000002.jpg' in names


def test_favorites_empty(db_client):
    response = db_client.get('/favorites')
    assert response.status_code == 200
    assert response.json() == {'favorites': []}
