import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

MOCK_METADATA = pd.DataFrame({
    'file_name': ['000001.jpg', '000002.jpg'],
    'labels': ["['person']", "['car']"],
})
MOCK_CLUSTER_LABELS = np.array([0, 1])
MOCK_OUTLIER_SCORES = np.array([0.1, 0.9])


@pytest.fixture(scope='module')
def client():
    with (
        patch('api.main.load_model', return_value=(MagicMock(), MagicMock(), MagicMock(), 'cpu')),
        patch('api.main.load_index', return_value=(MagicMock(), MOCK_METADATA)),
        patch('api.main.load_curation_data', return_value=(MOCK_CLUSTER_LABELS, MOCK_OUTLIER_SCORES)),
        patch('api.main.get_cluster_choices', return_value=[('Cluster 0 — person (2 images)', 0)]),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c
