# RAG-backed support agent

Answers questions over a corpus of IT incident tickets via a LangGraph agent with two tools,
citations, and honest "I don't know" behavior when retrieval comes up empty.

## Corpus note

The assignment's default corpus is ~30 pages of product docs plus a CSV of mock tickets. I was
given a single Excel file instead — 118 rows, 23 columns, ServiceNow-style IT incident tickets
for a cruise line's shipboard IT systems (ship codes, categories like Hardware/Application/
Security/Database, short description + description + resolution notes per ticket). I'm using
this as the entire corpus rather than sourcing separate product docs, since each ticket is
already a complete, citable unit of information.

## Stack

- FastAPI, fully async route handlers
- LangGraph for the agent
- Chroma (local, persisted to disk) as the vector store
- OpenAI-compatible gateway for chat (`gpt-5-mini`) and embeddings (`text-embedding-3-large`)
- pandas for structured ticket data
- Config entirely from environment variables via `pydantic-settings`

## Chunking strategy

**One chunk per ticket, not fixed-size text splitting.** Each row is already a semantically
atomic unit — a problem description paired with its resolution. Splitting further (e.g. a
generic recursive character splitter) would risk separating a description from its resolution,
which is exactly the connection a support agent needs when answering "has this happened
before, and how was it fixed."

Chunk text template:
```
Ticket {Number} — {Category}, {Priority}, {Location}
{Short description}
{Description}
Resolution: {Resolution notes}
```

Metadata stored alongside each chunk: `category`, `priority`, `location`, `assignment_group`,
`incident_state`, `resolution_code`, `configuration_item`, `impact`.

Fields deliberately excluded from both chunk text and metadata: `Severity` (constant across
all 118 rows — no signal), `Active`/`State` (redundant with `Incident state`, confirmed
identical across all rows), `Work notes` (internal analyst scratch notes, not resolution
content), and timestamp/person-name fields (`Created`, `Resolved`, `Updated`, `Assigned to`,
`Resolved by`, `Created by`, `Opened by`) since no question in scope needs "who" or "when."

## Deduplication / re-ingestion

Chunk ID = the ticket's `Number` field (e.g. `INC0420646`), already a unique, stable
identifier — no content hashing needed. Upsert-by-ID means re-ingesting the same file
overwrites matching tickets in place rather than duplicating them, and also means an updated
resolution on an existing ticket is picked up correctly on re-ingest (a hash-based ID would
instead treat an edited ticket as a brand-new chunk).

`POST /ingest` reports `chunks_added` vs `chunks_updated` separately (checked against existing
IDs before upsert) so the no-duplication behavior is demonstrable, not just asserted.

## The two tools

- **`search_tickets(query)`** — vector similarity search over the ticket corpus, for "has
  anything like X happened before" style questions.
- **`lookup_ticket(number, category, location, priority, incident_state, limit)`** — structured
  filter over the same underlying data, for exact-ID or aggregate questions ("what's the status
  of INC0420646", "how many critical Security incidents on RP"). Returns `total_count`
  separately from the (possibly truncated) `results` list, so "how many" questions are answered
  from a real count rather than by counting a limited results page.

Both tools read the same dataset but serve genuinely different query patterns, so the agent
has to make a real routing decision rather than picking between two near-identical tools.

## Retrieval / empty-retrieval handling

Chroma is configured for cosine distance; similarity = `1 - distance`. `search_tickets` checks
each match against `SIMILARITY_THRESHOLD` (default `0.35`) and returns an explicit
`NO_RELEVANT_CONTEXT` sentinel when nothing clears it, rather than handing the LLM weak matches
to paper over. The agent's system prompt instructs it to say it doesn't know when it receives
that sentinel (or an empty `lookup_ticket` result), instead of falling back on general
knowledge or guessing.

## Where I think it breaks

- **`gpt-5-mini` is a reasoning model** — it spends a variable number of hidden reasoning
  tokens before producing visible output, drawn from the same `max_tokens` budget. A too-tight
  `max_tokens` silently truncates to an empty answer with no error. Mitigated with a generous
  default (`MAX_TOKENS=1500`), but under-tested at the edges.
- **No content-hash diff on ingest** — every `/ingest` call re-embeds and re-upserts all rows,
  even unchanged ones. Fine at 118 rows (one batch embedding call); at 50x scale this would be
  the first optimization.
- **Chroma's Python client is synchronous** under the hood; the async wrapper functions call it
  directly rather than offloading to a thread pool. Non-issue at 118 rows (in-process, no
  network), but at real scale this would briefly block the event loop.
- **The agent can reformulate a failing query multiple times** before answering, rather than
  accepting one `NO_RELEVANT_CONTEXT` result as final — observed up to 3-4 reformulations on
  ambiguous off-corpus questions before a `recursion_limit` safety net (with a graceful
  fallback answer) catches it. This adds latency and, in at least one observed case, led the
  model to speculate about company departments not present in the corpus when a weak-but-real
  match came back instead of a clean empty result. A stricter prompt rule (stop after one
  empty result; never speculate beyond what tool results contain) is the fix, not yet applied.
- **Threshold-clearing weak matches, not true emptiness, is the more common failure mode in
  practice.** Several off-corpus test questions ("refund policy," "crew overtime policy") found
  loosely-related tickets that cleared the similarity threshold rather than returning nothing —
  the model handled these honestly in substance, but it means the clean "corpus truly has
  nothing" path is proven via a deterministic mocked test, not yet via a live example that
  reliably triggers it end-to-end.

## Tests

5 tests in `tests/`: tool-routing in both directions (calls a tool / answers directly),
ingest no-duplication, a `/query` happy path, and the required empty-retrieval case
(`test_query_empty_retrieval_answers_honestly`), which forces an empty vector-store result
deterministically via mocking rather than depending on a live question that happens to find no
corpus match.

```bash
pytest tests/ -v
```

## Known Limitations - What I'd fix or build next, given another week

First, the two correctness issues already identified: a stricter system-prompt rule so the
agent stops after one empty retrieval instead of reformulating repeatedly, and the same
prompt-level guardrail against speculating about anything not present in tool results — both
observed as real failure modes during testing, not just theoretical risks. Alongside that, I'd
find a live question that reliably triggers true empty retrieval, so that behavior is
demonstrated end-to-end through the running service rather than only through a mocked test.

After that, retrieval quality itself: hybrid search (BM25 + embeddings) would likely help on
short, keyword-heavy queries like exact ticket numbers or category names, where pure semantic
search is overkill and adds latency for no benefit. A content-hash diff on ingest would remove
the current "re-embed everything every time" cost, which matters far more at 50x the row count
than at 118. I'd also move the Chroma calls off the request thread, add token/cost logging per
request since latency alone doesn't say what a query actually cost, and add a real content-diff
test asserting that an updated resolution note is correctly picked up on re-ingest rather than
just claimed in the README.

Lower priority: the chat UI (built as a separate Streamlit app, not required by the brief) is
functional but plain — a next pass would add conversation memory across turns and nicer
citation rendering.

## Setup

```bash
git clone <repo-url> && cd RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and OPENAI_BASE_URL
uvicorn app.main:app --reload
```

Ingest the corpus (defaults to the bundled ticket file if no upload is given):
```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Ask a question:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "have stateroom TVs gone black before?"}'
```

Stream a question (Swagger's "Try it out" can't render SSE progressively — use `curl -N`, or
`http://127.0.0.1:8000/docs` for `/ingest` and `/query`):
```bash
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "have stateroom TVs gone black before?"}'
```

Run the tests (requires `pytest.ini` at the repo root with `[pytest]\nasyncio_mode = auto`):
```bash
pytest tests/ -v
```

### Optional: chat UI

A separate Streamlit app is included for interactively demoing the service (not required by
the assignment). It's a client of the API, not embedded in it — run it alongside the server,
in a second terminal, with the API already running:
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501` and talks to the API at `http://127.0.0.1:8000`.
