import io
import numpy as np
import pytest
from unittest.mock import patch
from PIL import Image as PILImage

from api.schemas import SearchRequest, SearchResponse, ImageResult
from tests.conftest import MOCK_METADATA


def test_search_request_defaults():
    req = SearchRequest(query='a dog')
    assert req.k == 5


def test_search_request_custom_k():
    req = SearchRequest(query='a cat', k=10)
    assert req.k == 10


def test_search_response_structure():
    resp = SearchResponse(results=[ImageResult(file_name='001.jpg', score=0.9)])
    assert resp.results[0].file_name == '001.jpg'
    assert resp.results[0].score == 0.9


def test_app_starts(client):
    response = client.get('/docs')
    assert response.status_code == 200


def test_search_returns_results(client):
    mock_results = MOCK_METADATA.copy()
    mock_results['score'] = [0.9, 0.8]

    with patch('api.main.encode_text', return_value=np.zeros((1, 512))), \
         patch('api.main.search', return_value=mock_results):
        response = client.post('/search', json={'query': 'a dog', 'k': 2})

    assert response.status_code == 200
    data = response.json()
    assert len(data['results']) == 2
    assert data['results'][0]['file_name'] == '000001.jpg'
    assert data['results'][0]['score'] == pytest.approx(0.9)


def test_search_respects_k(client):
    mock_results = MOCK_METADATA.iloc[:1].copy()
    mock_results['score'] = [0.9]

    with patch('api.main.encode_text', return_value=np.zeros((1, 512))), \
         patch('api.main.search', return_value=mock_results):
        response = client.post('/search', json={'query': 'a dog', 'k': 1})

    assert len(response.json()['results']) == 1


def test_outliers_all_categories(client):
    response = client.post('/outliers', json={'top_n': 2, 'category': 'All'})
    assert response.status_code == 200
    data = response.json()
    assert len(data['results']) == 2
    assert data['results'][0]['score'] == pytest.approx(0.9)


def test_outliers_filtered_category(client):
    response = client.post('/outliers', json={'top_n': 5, 'category': 'person'})
    assert response.status_code == 200
    data = response.json()
    assert len(data['results']) == 1
    assert data['results'][0]['file_name'] == '000001.jpg'


def test_get_clusters(client):
    response = client.get('/clusters')
    assert response.status_code == 200
    data = response.json()
    assert len(data['clusters']) == 1
    assert data['clusters'][0]['cluster_id'] == 0


def test_browse_cluster(client):
    response = client.post('/clusters/browse', json={'cluster_id': 0, 'k': 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data['results']) >= 1
    assert 'file_name' in data['results'][0]


def test_label_returns_predictions(client):
    img = PILImage.new('RGB', (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    mock_results = MOCK_METADATA.copy()
    mock_results['score'] = [0.9, 0.8]
    mock_labels = [
        {'label': 'person', 'confidence': 0.7},
        {'label': 'car', 'confidence': 0.3},
    ]

    with patch('api.main.encode_image', return_value=np.zeros((1, 512))), \
         patch('api.main.search', return_value=mock_results), \
         patch('api.main.pseudo_label', return_value=mock_labels):
        response = client.post('/label', files={'file': ('test.jpg', buf, 'image/jpeg')})

    assert response.status_code == 200
    data = response.json()
    assert data['labels'][0]['label'] == 'person'
    assert data['labels'][0]['confidence'] == pytest.approx(0.7)
    assert len(data['similar_images']) == 2
