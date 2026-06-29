from datetime import datetime

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ImageResult(BaseModel):
    file_name: str
    score: float


class SearchResponse(BaseModel):
    results: list[ImageResult]


class LabelResult(BaseModel):
    label: str
    confidence: float


class LabelResponse(BaseModel):
    labels: list[LabelResult]
    similar_images: list[ImageResult]


class ClusterBrowseRequest(BaseModel):
    cluster_id: int
    k: int = 20


class ClusterChoice(BaseModel):
    label: str
    cluster_id: int


class ClustersResponse(BaseModel):
    clusters: list[ClusterChoice]


class OutliersRequest(BaseModel):
    top_n: int = 20
    category: str = 'All'


class SearchHistoryItem(BaseModel):
    id: int
    query: str
    result_count: int
    timestamp: datetime


class HistoryResponse(BaseModel):
    history: list[SearchHistoryItem]


class FavoriteItem(BaseModel):
    id: int
    file_name: str
    timestamp: datetime


class FavoritesResponse(BaseModel):
    favorites: list[FavoriteItem]
