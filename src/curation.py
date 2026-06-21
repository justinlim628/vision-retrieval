import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import N_CLUSTERS


def _minmax(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def load_curation_data(embeddings_path: Path, index, metadata: pd.DataFrame):
    embeddings = np.load(str(embeddings_path)).astype(np.float32)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(embeddings)

    centroid_dist = np.linalg.norm(embeddings - km.cluster_centers_[cluster_labels], axis=1)

    nn_scores, _ = index.search(embeddings, 11)
    mean_nn_sim = nn_scores[:, 1:].mean(axis=1)

    outlier_scores = 0.5 * _minmax(centroid_dist) + 0.5 * (1 - _minmax(mean_nn_sim))

    return cluster_labels, outlier_scores


def get_cluster_choices(cluster_labels: np.ndarray, metadata: pd.DataFrame) -> list:
    choices = []
    for cid in range(N_CLUSTERS):
        mask = cluster_labels == cid
        rows = metadata[mask]
        all_labels = [
            lbl
            for labels_str in rows['labels']
            for lbl in ast.literal_eval(labels_str)
        ]
        dominant = Counter(all_labels).most_common(1)[0][0] if all_labels else 'unknown'
        choices.append((f'Cluster {cid} — {dominant} ({mask.sum()} images)', cid))
    return choices
