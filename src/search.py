import faiss
import pandas as pd
import numpy as np
from pathlib import Path


def load_index(index_path: Path, metadata_path: Path):
    index = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path)
    return index, metadata


def search(query_embedding: np.ndarray, index, metadata: pd.DataFrame, k: int = 9):
    scores, indices = index.search(query_embedding, k)

    results = metadata.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results
