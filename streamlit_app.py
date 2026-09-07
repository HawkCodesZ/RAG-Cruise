import json
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Incident Desk", page_icon="⚓", layout="centered")

st.markdown(
    "<h3 style='font-family: monospace; margin-bottom: 0;'>⚓ Incident Desk</h3>"
    "<p style='color: #5C6B7A; font-size: 13px; margin-top: 2px;'>"
    "118 tickets on file · shipboard IT</p>",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str, "sources": [...]}

# --- render existing turns ---
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            chips = " ".join(f"`{s['chunk_id']}`" for s in turn["sources"])
            st.caption(chips)


def stream_query(question: str):
    """Yields (event_type, data_dict) tuples parsed from the SSE response."""
    with requests.post(
        f"{API_BASE}/query/stream",
        json={"question": question},
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        event_type, data_line = None, None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            if raw_line.startswith("event: "):
                event_type = raw_line[len("event: "):].strip()
            elif raw_line.startswith("data: "):
                data_line = raw_line[len("data: "):]
            elif raw_line == "" and event_type and data_line:
                yield event_type, json.loads(data_line)
                event_type, data_line = None, None


question = st.chat_input("Ask the desk…")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        answer_text = ""
        sources = []
        tool_used = None

        try:
            for event_type, data in stream_query(question):
                if event_type == "status":
                    status_placeholder.caption(f"› {data['message']}")
                elif event_type == "token":
                    answer_text += data["content"]
                    answer_placeholder.markdown(answer_text + "▌")
                elif event_type == "done":
                    status_placeholder.empty()
                    sources = data.get("sources", [])
                    tool_used = data.get("tool_used")
                    if not answer_text.strip():
                        answer_text = "I don't have information on that in the incident ticket corpus."
                    answer_placeholder.markdown(answer_text)
        except requests.exceptions.RequestException:
            status_placeholder.empty()
            answer_text = "Connection to the desk was interrupted. Try again."
            answer_placeholder.error(answer_text)

        if sources:
            chips = " ".join(f"`{s['chunk_id']}`" for s in sources)
            st.caption(chips)

    st.session_state.history.append({
        "role": "assistant", "content": answer_text, "sources": sources,
    })