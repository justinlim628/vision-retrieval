import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_api_base(monkeypatch):
    monkeypatch.setenv('API_BASE_URL', 'http://localhost:8000')


def test_search_images_returns_results():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        'results': [
            {'file_name': '000001.jpg', 'score': 0.9},
            {'file_name': '000002.jpg', 'score': 0.8},
        ]
    }
    with patch('httpx.post', return_value=mock_resp):
        from mcp_server.server import search_images
        result = search_images('a dog', k=2)
    assert len(result) == 2
    assert result[0]['file_name'] == '000001.jpg'
    assert result[0]['score'] == pytest.approx(0.9)


def test_search_images_passes_query_and_k():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {'results': []}
    with patch('httpx.post', return_value=mock_resp) as mock_post:
        from mcp_server.server import search_images
        search_images('a cat', k=3)
    mock_post.assert_called_once_with(
        'http://localhost:8000/search',
        json={'query': 'a cat', 'k': 3},
    )


def test_get_favorites_returns_list():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        'favorites': [{'id': 1, 'file_name': '000001.jpg', 'timestamp': '2024-01-01T00:00:00'}]
    }
    with patch('httpx.get', return_value=mock_resp):
        from mcp_server.server import get_favorites
        result = get_favorites()
    assert len(result) == 1
    assert result[0]['file_name'] == '000001.jpg'


def test_get_favorites_empty():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {'favorites': []}
    with patch('httpx.get', return_value=mock_resp):
        from mcp_server.server import get_favorites
        result = get_favorites()
    assert result == []


def test_add_favorite_returns_item():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        'id': 1, 'file_name': '000001.jpg', 'timestamp': '2024-01-01T00:00:00'
    }
    with patch('httpx.post', return_value=mock_resp):
        from mcp_server.server import add_favorite
        result = add_favorite('000001.jpg')
    assert result['file_name'] == '000001.jpg'
