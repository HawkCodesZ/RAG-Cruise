from functools import lru_cache
from openai import AsyncOpenAI

from app.config import get_settings

EMBEDDING_BATCH_SIZE = 20


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=60.0,  # explicit — fail loudly in a minute rather than hang
        max_retries=2,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed in batches so a single large request doesn't risk timing out
    the whole call. Note: if a batch fails partway through, the exception
    propagates and any already-embedded batches in this call are discarded —
    there's no partial-progress persistence. Fine for a single ingest run at
    118 rows; would need reworking if partial-failure recovery becomes
    necessary at larger scale."""
    settings = get_settings()
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        resp = await client.embeddings.create(model=settings.embedding_model, input=batch)
        all_embeddings.extend(item.embedding for item in resp.data)

    return all_embeddings


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def chat_completion(messages: list[dict], **kwargs) -> str:
    """Plain (non-streaming) chat completion, returns the text content."""
    settings = get_settings()
    client = get_client()
    resp = await client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        max_tokens=kwargs.pop("max_tokens", settings.max_tokens),
        **kwargs,
    )
    return resp.choices[0].message.content or ""