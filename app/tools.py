import json

from langchain_core.tools import tool

from app.config import get_settings
from app.vectorstore import query_similar, get_collection

NO_CONTEXT_SENTINEL = "NO_RELEVANT_CONTEXT"


# --- Core logic (plain async functions, directly unit-testable) ---

async def search_tickets_impl(query: str) -> dict:
    """Semantic search over the ticket corpus. Returns matches above the
    similarity threshold, or an explicit sentinel when nothing clears it —
    this is the mechanism behind 'say I don't know rather than guess'."""
    settings = get_settings()
    raw = await query_similar(query, top_k=settings.top_k)

    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    distances = raw["distances"][0]

    matches = []
    for id_, doc, meta, dist in zip(ids, docs, metas, distances):
        similarity = 1 - dist  # cosine distance -> similarity
        if similarity >= settings.similarity_threshold:
            matches.append({
                "ticket_id": id_,
                "similarity": round(similarity, 3),
                "text": doc,
                "metadata": meta,
            })

    if not matches:
        return {"found": False, "sentinel": NO_CONTEXT_SENTINEL, "matches": []}

    return {"found": True, "matches": matches}


def _build_where(category=None, location=None, priority=None, incident_state=None) -> dict | None:
    clauses = []
    if category:
        clauses.append({"category": category})
    if location:
        clauses.append({"location": location})
    if priority:
        clauses.append({"priority": priority})
    if incident_state:
        clauses.append({"incident_state": incident_state})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


async def lookup_ticket_impl(
    number: str | None = None,
    category: str | None = None,
    location: str | None = None,
    priority: str | None = None,
    incident_state: str | None = None,
    limit: int = 10,
) -> dict:
    collection = get_collection()

    if number:
        result = collection.get(ids=[number], include=["documents", "metadatas"])
        if not result["ids"]:
            return {"found": False, "results": []}
        return {
            "found": True,
            "results": [{
                "ticket_id": result["ids"][0],
                "text": result["documents"][0],
                "metadata": result["metadatas"][0],
            }],
        }

    where = _build_where(category, location, priority, incident_state)
    # Fetch all matches first so we know the true count, then slice for display.
    all_matches = collection.get(where=where, include=["documents", "metadatas"])
    total_count = len(all_matches["ids"])

    if total_count == 0:
        return {"found": False, "results": [], "total_count": 0}

    ids = all_matches["ids"][:limit]
    docs = all_matches["documents"][:limit]
    metas = all_matches["metadatas"][:limit]

    return {
        "found": True,
        "total_count": total_count,
        "returned_count": len(ids),
        "truncated": total_count > limit,
        "results": [
            {"ticket_id": i, "text": d, "metadata": m}
            for i, d, m in zip(ids, docs, metas)
        ],
    }


# --- LangGraph-facing tool wrappers ---
# Return JSON strings, not dicts — tool output becomes a ToolMessage's
# content, which should be text the LLM can read reliably.

@tool
async def search_tickets(query: str) -> str:
    """Semantically search past incident tickets for similar issues.
    Use this when the user asks whether something like their problem
    has happened before, or wants examples of similar incidents."""
    result = await search_tickets_impl(query)
    return json.dumps(result)


@tool
async def lookup_ticket(
    number: str | None = None,
    category: str | None = None,
    location: str | None = None,
    priority: str | None = None,
    incident_state: str | None = None,
    limit: int = 10,
) -> str:
    """Look up a specific ticket by its number, or filter tickets by exact
    category/location/priority/incident_state. Returns total_count separately
    from the (possibly truncated) results list — use total_count for 'how many'
    questions, don't count the results list yourself, it may be truncated."""
    result = await lookup_ticket_impl(
        number=number, category=category, location=location,
        priority=priority, incident_state=incident_state, limit=limit,
    )
    return json.dumps(result)