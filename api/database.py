from datetime import datetime, timezone
from typing import Optional, Generator

from sqlmodel import Field, Session, SQLModel, create_engine

from src.config import DB_PATH


class SearchHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    result_count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Favorite(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_name: str = Field(index=True, unique=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


engine = create_engine(DB_PATH, connect_args={'check_same_thread': False})


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
