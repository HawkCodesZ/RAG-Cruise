from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    xlsx_path: str | None = Field(
        default=None,
        description="Path to the tickets xlsx. Defaults to settings.tickets_xlsx_path if omitted.",
    )


class IngestResponse(BaseModel):
    document_id: str
    chunks_added: int
    chunks_updated: int
    chunks_skipped: int
    total_chunks_in_store: int


class SourceChunk(BaseModel):
    document_name: str
    chunk_id: str
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    tool_used: str | None = None  # "search_tickets" | "lookup_ticket" | None (direct answer)