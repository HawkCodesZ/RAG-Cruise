from functools import lru_cache
import chromadb

from app.config import get_settings
from app.llm import embed_texts

import json
from pathlib import Path


@lru_cache
def get_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    # No embedding_function passed — we embed ourselves via the gateway
    # and hand Chroma raw vectors, since Chroma's built-in OpenAI wrapper
    # doesn't cleanly support a custom base_url across versions.
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


async def upsert_chunks(ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    """Embed and upsert. Upsert-by-id means re-ingesting an existing
    ticket ID overwrites in place rather than duplicating."""
    if not ids:
        return
    embeddings = await embed_texts(documents)
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


async def query_similar(query_text: str, top_k: int) -> dict:
    embedding = (await embed_texts([query_text]))[0]
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

def export_snapshot(path: str | None = None) -> None:
    """Dump the full current state of the collection to a JSON file, so the
    ingested corpus can be inspected without querying Chroma directly —
    useful for demos and the review call. Overwrites on every call, so it
    always reflects the current store, not a history of past states."""
    settings = get_settings()
    snapshot_path = path or settings.chunks_snapshot_path

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    snapshot = [
        {"ticket_id": id_, "text": doc, "metadata": meta}
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)


def collection_count() -> int:
    return get_collection().count()