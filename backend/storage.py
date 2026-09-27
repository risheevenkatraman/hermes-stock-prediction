"""Shared application storage. PostgreSQL in AWS, SQLite for local development."""

from functools import lru_cache
import os
from pathlib import Path

from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine

metadata = MetaData()
workspaces = Table(
    "workspaces",
    metadata,
    Column("user_id", String(128), primary_key=True),
    Column("payload", JSON, nullable=False),
)
research = Table(
    "research_snapshots",
    metadata,
    Column("ticker", String(10), primary_key=True),
    Column("payload", JSON, nullable=False),
)


@lru_cache(maxsize=4)
def engine_for(url: str):
    if url.startswith("sqlite"):
        Path("data").mkdir(exist_ok=True)
    return create_engine(url, pool_pre_ping=True)


def engine():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        if os.getenv("HERMES_DEPLOYMENT") == "production":
            raise RuntimeError("DATABASE_URL is required in production")
        url = "sqlite:///data/hermes.sqlite3"
    return engine_for(url)


def initialize():
    metadata.create_all(engine())


def read_record(table, key: str):
    with engine().connect() as connection:
        return (
            connection.execute(table.select().where(list(table.primary_key)[0] == key))
            .mappings()
            .first()
        )


def save_record(table, key: str, payload: dict):
    # Atomic upsert avoids a select/insert race on first account save or job retry.
    db = engine()
    if db.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    primary = list(table.primary_key)[0]
    statement = insert(table).values({primary.name: key, "payload": payload})
    statement = statement.on_conflict_do_update(
        index_elements=[primary], set_={"payload": payload}
    )
    with db.begin() as connection:
        connection.execute(statement)


if __name__ == "__main__":
    initialize()
    print("Hermes application tables initialized.")
