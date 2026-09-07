import pytest
from unittest.mock import AsyncMock, MagicMock


def test_query_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.run_agent",
        AsyncMock(return_value=("INC0000001 covers this.", [], "search_tickets")),
    )
    response = client.post("/query", json={"question": "Have TVs gone black before?"})
    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "search_tickets"
    assert body["answer"] == "INC0000001 covers this."
    assert "latency_ms" in body


def test_query_empty_retrieval_answers_honestly(client, monkeypatch, mock_embed, mock_collection):
    """The required empty-retrieval case: when the vector store has no match
    clearing the similarity threshold, the tool must return the
    NO_RELEVANT_CONTEXT sentinel, and the agent's system prompt instructs it
    to say so honestly rather than guess. This test forces that condition
    deterministically via the mocked vector store, rather than relying on
    finding a real-world question the live model happens to fail to answer."""
    

    mock_collection.query.return_value = {
        "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]],
    }
    monkeypatch.setattr("app.vectorstore.get_collection", lambda: mock_collection)

    from langchain_core.messages import AIMessage
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "search_tickets", "args": {"query": "unrelated topic"}, "id": "call_1"}],
    )
    final_response = AIMessage(
        content="I don't have information on that in the incident ticket corpus."
    )
    import app.agent as agent_module
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    response = client.post("/query", json={"question": "unrelated topic with no corpus match"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "don't" in body["answer"].lower() or "no information" in body["answer"].lower()