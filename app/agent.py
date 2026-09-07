import json

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from app.config import get_settings
from app.tools import search_tickets, lookup_ticket
from app.models import SourceChunk

SYSTEM_PROMPT = """You are a support agent answering questions about a corpus \
of IT incident tickets from a cruise line's shipboard systems.

You have two tools:
- search_tickets: semantic search, for "has something like this happened \
before" style questions.
- lookup_ticket: exact ticket-number lookup, or structured filtering by \
category/location/priority/incident_state.

Rules:
- Call search_tickets or lookup_ticket for any question that could plausibly relate to IT
  incidents, shipboard systems, or company operations, even if you're unsure the corpus
  covers it. Only skip tool calls for unambiguous general-knowledge questions that have
  nothing to do with a cruise line's incident tickets (e.g. math, geography, trivia).
- If a tool result contains "NO_RELEVANT_CONTEXT" or found: false, say plainly that the
  corpus doesn't contain information on that topic. Do not guess, do not offer to fetch
  the information from elsewhere, and do not ask clarifying questions to route around it —
  a flat, honest "I don't know" is the correct answer here.
- When you do use ticket information, cite the ticket number(s) in your answer.
- When answering "how many" questions, use the tool's total_count field, not the
  number of items in the results list, which may be truncated.
- Never invent or speculate about company departments, policies, contacts, or processes
  that are not present in the tool results you received. If you don't have grounded
  information to answer part of a question, say so plainly rather than suggesting who
  might have it.
- When a tool result includes a total_count field, use that number for "how many"
  questions — never count the items in a results list yourself, since it may be
  truncated to fewer than the true total.
- Be concise."""

_settings = get_settings()

_llm = ChatOpenAI(
    model=_settings.chat_model,
    api_key=_settings.openai_api_key,
    base_url=_settings.openai_base_url,
    max_tokens=_settings.max_tokens,
).bind_tools([search_tickets, lookup_ticket])


# async def agent_node(state: MessagesState):
#     messages = state["messages"]
#     if not any(isinstance(m, SystemMessage) for m in messages):
#         messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
#     response = await _llm.ainvoke(messages)
#     return {"messages": [response]}

async def agent_node(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = await _llm.ainvoke(messages, config)
    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            print(f"[tool_call] {tc['name']}({tc['args']})")
    return {"messages": [response]}


_tool_node = ToolNode([search_tickets, lookup_ticket])

_builder = StateGraph(MessagesState)
_builder.add_node("agent", agent_node)
_builder.add_node("tools", _tool_node)
_builder.set_entry_point("agent")
_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
_builder.add_edge("tools", "agent")

graph = _builder.compile()


def _extract_sources_and_tool(messages) -> tuple[list[SourceChunk], str | None]:
    sources: list[SourceChunk] = []
    seen_ids: set[str] = set()
    tool_used: str | None = None

    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_used = m.tool_calls[0]["name"]
        if isinstance(m, ToolMessage):
            try:
                payload = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                continue
            items = payload.get("matches") or payload.get("results") or []
            for item in items:
                if item["ticket_id"] in seen_ids:
                    continue
                seen_ids.add(item["ticket_id"])
                sources.append(SourceChunk(
                    document_name="incident_tickets",
                    chunk_id=item["ticket_id"],
                    snippet=item["text"][:200],
                ))
    return sources, tool_used


async def run_agent(question: str) -> tuple[str, list[SourceChunk], str | None]:
    # result = await graph.ainvoke(
    #             {"messages": [HumanMessage(content=question)]},
    #             config={"recursion_limit": 8},
    #         )
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 8},
        )
    except GraphRecursionError:
        return (
            "I wasn't able to find a clear answer to that in the incident ticket corpus.",
            [],
            None,
        )
    messages = result["messages"]
    answer = messages[-1].content
    sources, tool_used = _extract_sources_and_tool(messages)
    return answer, sources, tool_used