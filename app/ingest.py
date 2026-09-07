import pandas as pd

from app.config import get_settings
from app.models import IngestResponse
from app.vectorstore import upsert_chunks, get_collection, collection_count, export_snapshot

# Constant across all 118 rows (Severity is always "3 - Low") — no signal,
# so it's excluded from both the embedding text and metadata.
#EXCLUDED_FIELDS = {"Severity", "Active", "Work notes"}


def _safe(value) -> str:
    """pandas gives NaN for empty cells; normalize to empty string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _build_chunk_text(row: pd.Series) -> str:
    """One chunk per ticket. We deliberately don't split a ticket's
    description from its resolution — they're one semantic unit, and
    splitting them would break exactly the connection a support agent
    needs to answer 'has this happened before, and how was it fixed'."""
    return (
        f"Ticket {row['Number']} — {_safe(row['Category'])}, "
        f"{_safe(row['Priority'])}, {_safe(row['Location'])}\n"
        f"{_safe(row['Short description'])}\n"
        f"{_safe(row['Description'])}\n"
        f"Resolution: {_safe(row['Resolution notes'])}"
    )


def _build_metadata(row: pd.Series) -> dict:
    return {
        "category": _safe(row["Category"]),
        "priority": _safe(row["Priority"]),
        "location": _safe(row["Location"]),
        "assignment_group": _safe(row["Assignment group"]),
        "incident_state": _safe(row["Incident state"]),
        "resolution_code": _safe(row["Resolution code"]),
        "configuration_item": _safe(row["Configuration item"]),
        "impact": _safe(row["Impact"]),
    }


async def ingest_tickets(xlsx_path: str | None = None) -> IngestResponse:
    settings = get_settings()
    path = xlsx_path or settings.tickets_xlsx_path

    df = pd.read_excel(path)
    df["Number"] = df["Number"].astype(str)

    ids = df["Number"].tolist()
    documents, metadatas = [], []
    for _, row in df.iterrows():
        documents.append(_build_chunk_text(row))
        metadatas.append(_build_metadata(row))
    # documents = [_build_chunk_text(row) for _, row in df.iterrows()]
    # metadatas = [_build_metadata(row) for _, row in df.iterrows()]

    # Determine which ids already exist so we can report added vs updated,
    # not just "upserted N rows" — this is what makes the no-duplication
    # requirement demonstrable rather than just asserted.
    existing = get_collection().get(ids=ids, include=[])
    existing_ids = set(existing["ids"])

    added = sum(1 for i in ids if i not in existing_ids)
    updated = sum(1 for i in ids if i in existing_ids)

    

    await upsert_chunks(ids=ids, documents=documents, metadatas=metadatas)
    export_snapshot()
    return IngestResponse(
        document_id="incident_tickets",
        chunks_added=added,
        chunks_updated=updated,
        chunks_skipped=0,
        total_chunks_in_store=collection_count(),
    )
