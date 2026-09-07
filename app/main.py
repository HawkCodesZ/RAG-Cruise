import time
import tempfile
from pathlib import Path
import logging
import json
from typing import AsyncIterator

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from app.agent import graph
from app.models import SourceChunk

from fastapi import FastAPI, UploadFile, File, HTTPException, logger

from app.config import get_settings
from app.ingest import ingest_tickets
from app.models import IngestResponse
from app.agent import run_agent
from app.models import QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-backed support agent")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile | None = File(default=None)) -> IngestResponse:
    """Ingest the ticket corpus. Accepts an uploaded .xlsx file; if omitted,
    falls back to settings.tickets_xlsx_path (the default corpus already
    on disk). Re-ingesting is safe — chunk IDs are the ticket Number field,
    so matching tickets are upserted in place rather than duplicated."""
    xlsx_path: str | None = None

    if file is not None:
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="File must be a .xlsx spreadsheet.")
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(await file.read())
            xlsx_path = tmp.name

    try:
        result = await ingest_tickets(xlsx_path=xlsx_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ticket file not found: {xlsx_path or get_settings().tickets_xlsx_path}")
    finally:
        if xlsx_path is not None:
            Path(xlsx_path).unlink(missing_ok=True)

    return result

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question over the ticket corpus. The agent decides whether
    to call search_tickets, lookup_ticket, both, or neither."""
    start = time.perf_counter()
    try:
        answer, sources, tool_used = await run_agent(request.question)
    except Exception:
        logger.exception("Failed to process query: %r", request.question)
        raise HTTPException(status_code=500, detail="Failed to process the query.")
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(latency_ms, 2),
        tool_used=tool_used,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_agent_response(question: str) -> AsyncIterator[str]:
    seen_tool_names: set[str] = set()
    seen_ticket_ids: set[str] = set()
    sources: list[SourceChunk] = []
    tool_used: str | None = None

    try:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 8},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                name = event["name"]
                if tool_used is None:
                    tool_used = name
                if name not in seen_tool_names:
                    seen_tool_names.add(name)
                    yield _sse("status", {"message": f"Calling {name}..."})

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                content = getattr(output, "content", output)
                try:
                    payload = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                items = payload.get("matches") or payload.get("results") or []
                for item in items:
                    tid = item["ticket_id"]
                    if tid in seen_ticket_ids:
                        continue
                    seen_ticket_ids.add(tid)
                    sources.append(SourceChunk(
                        document_name="incident_tickets",
                        chunk_id=tid,
                        snippet=item["text"][:200],
                    ))

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse("token", {"content": chunk.content})

    except GraphRecursionError:
        yield _sse("token", {
            "content": "I wasn't able to find a clear answer to that in the incident ticket corpus."
        })

    yield _sse("done", {
        "sources": [s.model_dump() for s in sources],
        "tool_used": tool_used,
    })


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    return StreamingResponse(
        _stream_agent_response(request.question),
        media_type="text/event-stream",
    )