import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage

import app.agent as agent_module


@pytest.mark.asyncio
async def test_agent_calls_tool_for_incident_question(monkeypatch):
    """A question shaped like an incident lookup should trigger a tool call,
    not a direct answer."""
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "search_tickets", "args": {"query": "TV black screen"}, "id": "call_1"}],
    )
    final_response = AIMessage(content="Yes, INC0000001 covers this.")

    mock_ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
    fake_llm = MagicMock()
    fake_llm.ainvoke = mock_ainvoke
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    monkeypatch.setattr(
        "app.tools.search_tickets_impl",
        AsyncMock(return_value={"found": True, "matches": [
            {"ticket_id": "INC0000001", "similarity": 0.9, "text": "...", "metadata": {}}
        ]}),
    )

    answer, sources, tool_used = await agent_module.run_agent("Have TVs gone black before?")

    assert tool_used == "search_tickets"
    assert "INC0000001" in answer


@pytest.mark.asyncio
async def test_agent_skips_tool_for_general_knowledge(monkeypatch):
    """A general-knowledge question should be answered directly, no tool call."""
    direct_response = AIMessage(content="The capital of France is Paris.")
    mock_ainvoke = AsyncMock(return_value=direct_response)
    fake_llm = MagicMock()
    fake_llm.ainvoke = mock_ainvoke
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    answer, sources, tool_used = await agent_module.run_agent("What is the capital of France?")

    assert tool_used is None
    assert sources == []
    assert "Paris" in answer