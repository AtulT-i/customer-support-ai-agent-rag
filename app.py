import streamlit as st

from src.agent import SupportAgent, require_artifacts
from src.config import load_config


st.set_page_config(page_title="Spotify Support Agent", page_icon="🎧", layout="wide")
st.title("Spotify Support Agent")
st.caption(
    "Local accuracy-first demo: intent classification, historical evidence, "
    "and conservative escalation."
)

config = load_config()
use_ollama = st.sidebar.checkbox("Use local Ollama evidence selection", value=False)
ollama_model = st.sidebar.text_input("Installed Ollama model", value="qwen2.5:3b") if use_ollama else None
st.sidebar.caption("Default: local extractive RAG. Ollama may select evidence, never invent reply text.")
try:
    require_artifacts(config)
    agent = SupportAgent(config, ollama_model=ollama_model)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

message = st.text_area(
    "Incoming customer message",
    value="The app keeps pausing whenever I play a song.",
    height=100,
    max_chars=4000,
)

if st.button("Analyze message", type="primary"):
    try:
        with st.spinner("Retrieving local evidence..."):
            result = agent.respond(message)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    decision_color = "green" if result["decision"] == "auto_handle" else "orange"
    first, second, third = st.columns(3)
    first.metric("Intent", result["intent"].replace("_", " ").title())
    second.metric("Confidence", f"{result['confidence']:.1%}")
    third.markdown(
        f"**Decision**  \n:{decision_color}[{result['decision'].replace('_', ' ').title()}]"
    )

    st.markdown("### Draft reply")
    st.info(result["draft_reply"])
    st.markdown(f"**Decision reason:** {result['reason']}")
    st.caption(f"RAG mode: {result['rag_mode']} · Cited tweet IDs: {', '.join(result['evidence_tweet_ids']) or 'none'}")
    st.caption("Draft only: no reply is sent and no support ticket or account action is created.")

    st.markdown("### Historical evidence")
    for rank, item in enumerate(result["evidence"], start=1):
        with st.expander(
            f"#{rank} · similarity {item['similarity']:.3f} · "
            f"tweet {item['customer_tweet_id']}"
        ):
            st.markdown(f"**Customer:** {item['customer_text']}")
            st.markdown(f"**Brand reply:** {item['brand_reply']}")

    source = result["model_metadata"]["label_source"]
    if source != "human_reviewed":
        st.warning(
            "This model uses provisional weak labels. It is suitable for a "
            "pipeline demo, not for submission headline results."
        )
