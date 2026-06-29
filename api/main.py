import ast
import io
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from src.config import EMBEDDINGS_PATH, IMAGES_DIR, INDEX_PATH, LABEL_K, METADATA_PATH
from src.clip_model import encode_image, encode_text, load_model
from src.curation import get_cluster_choices, load_curation_data
from src.labeling import pseudo_label
from src.search import load_index, search

from api.database import Favorite, SearchHistory, create_db_and_tables, get_session
from api.schemas import (
    ClusterBrowseRequest,
    ClusterChoice,
    ClustersResponse,
    FavoriteItem,
    FavoritesResponse,
    HistoryResponse,
    ImageResult,
    LabelResponse,
    LabelResult,
    OutliersRequest,
    SearchHistoryItem,
    SearchRequest,
    SearchResponse,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading CLIP model...')
    state['model'], state['preprocess'], state['tokenizer'], state['device'] = load_model()
    print('Loading FAISS index...')
    state['index'], state['metadata'] = load_index(INDEX_PATH, METADATA_PATH)
    print('Computing curation data...')
    state['cluster_labels'], state['outlier_scores'] = load_curation_data(
        EMBEDDINGS_PATH, state['index'], state['metadata']
    )
    state['cluster_choices'] = get_cluster_choices(state['cluster_labels'], state['metadata'])
    create_db_and_tables()
    print('Ready.')
    yield


app = FastAPI(title='CLIP Image Search API', lifespan=lifespan)
app.mount('/images', StaticFiles(directory=str(IMAGES_DIR)), name='images')


@app.post('/label', response_model=LabelResponse)
async def label_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    embedding = encode_image(image, state['model'], state['preprocess'], state['device'])
    results = search(embedding, state['index'], state['metadata'], k=LABEL_K)
    label_results = pseudo_label(results)
    return LabelResponse(
        labels=[LabelResult(label=r['label'], confidence=r['confidence']) for r in label_results],
        similar_images=[
            ImageResult(file_name=row['file_name'], score=row['score'])
            for _, row in results.iterrows()
        ],
    )


@app.post('/outliers', response_model=SearchResponse)
def outliers_endpoint(req: OutliersRequest):
    if req.category == 'All':
        indices = np.argsort(state['outlier_scores'])[::-1][:req.top_n]
        rows = state['metadata'].iloc[indices]
        scores = state['outlier_scores'][indices]
    else:
        mask = state['metadata']['labels'].apply(
            lambda s: req.category in ast.literal_eval(s)
        )
        sub_meta = state['metadata'][mask]
        sub_scores = state['outlier_scores'][mask.values]
        order = np.argsort(sub_scores)[::-1][:req.top_n]
        rows = sub_meta.iloc[order]
        scores = sub_scores[order]
    return SearchResponse(
        results=[
            ImageResult(file_name=row['file_name'], score=float(score))
            for (_, row), score in zip(rows.iterrows(), scores)
        ]
    )


@app.get('/clusters', response_model=ClustersResponse)
def clusters_endpoint():
    return ClustersResponse(
        clusters=[
            ClusterChoice(label=label, cluster_id=cid)
            for label, cid in state['cluster_choices']
        ]
    )


@app.post('/clusters/browse', response_model=SearchResponse)
def browse_cluster(req: ClusterBrowseRequest):
    mask = state['cluster_labels'] == req.cluster_id
    rows = state['metadata'][mask]
    sample = rows.sample(n=min(req.k, len(rows)), random_state=0)
    return SearchResponse(
        results=[
            ImageResult(file_name=row['file_name'], score=1.0)
            for _, row in sample.iterrows()
        ]
    )


@app.post('/search', response_model=SearchResponse)
def search_endpoint(
    req: SearchRequest,
    session: Annotated[Session, Depends(get_session)],
):
    embedding = encode_text(req.query, state['model'], state['tokenizer'], state['device'])
    results = search(embedding, state['index'], state['metadata'], k=req.k)
    response = SearchResponse(
        results=[
            ImageResult(file_name=row['file_name'], score=row['score'])
            for _, row in results.iterrows()
        ]
    )
    session.add(SearchHistory(query=req.query, result_count=len(response.results)))
    session.commit()
    return response


@app.get('/history', response_model=HistoryResponse)
def history_endpoint(session: Annotated[Session, Depends(get_session)]):
    rows = session.exec(
        select(SearchHistory).order_by(SearchHistory.timestamp.desc(), SearchHistory.id.desc())
    ).all()
    return HistoryResponse(
        history=[SearchHistoryItem.model_validate(r, from_attributes=True) for r in rows]
    )


@app.post('/favorites/{file_name}', response_model=FavoriteItem)
def add_favorite(file_name: str, session: Annotated[Session, Depends(get_session)]):
    fav = Favorite(file_name=file_name)
    session.add(fav)
    try:
        session.commit()
        session.refresh(fav)
    except IntegrityError:
        session.rollback()
        fav = session.exec(select(Favorite).where(Favorite.file_name == file_name)).one()
    return FavoriteItem.model_validate(fav, from_attributes=True)


@app.get('/favorites', response_model=FavoritesResponse)
def list_favorites(session: Annotated[Session, Depends(get_session)]):
    rows = session.exec(select(Favorite).order_by(Favorite.timestamp.desc())).all()
    return FavoritesResponse(
        favorites=[FavoriteItem.model_validate(r, from_attributes=True) for r in rows]
    )
