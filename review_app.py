"""Separate single-reviewer UI. No model predictions or judge scores are prefilled."""
import json

import streamlit as st

from src.config import load_config
from src.human_scoring import save_human_score
from src.judge import RUBRIC_FIELDS
from src.submission import annotation_progress, read_table, save_annotation


st.set_page_config(page_title="Human review workspace", layout="wide")
st.title("Human review workspace")
st.warning("Single reviewer/editor at a time. Saves are explicit and backed up. Never mark AI-generated labels as human-reviewed.")
config = load_config()
mode = st.sidebar.radio("Review task", ["Intent and decision labels", "Reply quality scores"])
annotator = st.sidebar.text_input("Your name or annotator ID")
st.sidebar.caption("Do not use classifier confidence or judge scores as ground truth.")

if mode == "Intent and decision labels":
    path = config.data["label_queue_csv"]
    queue = read_table(path)
    progress = annotation_progress(queue, config.intents)
    st.info(f"{progress['reviewed_rows']} reviewed; {progress['usable_unique_thread_rows']} usable unique-thread rows. Minimum 300 for 200 golden + 100 training; 500+ recommended.")
    with st.expander("Progress and intent coverage"):
        st.json(progress)
    show_all = st.checkbox("Include already reviewed examples")
    remaining = queue if show_all else queue[~queue["reviewed"].eq("yes")]
    if remaining.empty:
        st.success("No pending examples in this view.")
        st.stop()
    example_id = st.selectbox("Example", remaining["example_id"].tolist())
    row = remaining[remaining["example_id"].eq(example_id)].iloc[0]
    st.subheader("Customer message")
    st.text(row["text"])
    with st.expander("Historical reply — context only, not ground truth"):
        st.text(row.get("historical_reply", ""))
    with st.form(f"label-{example_id}"):
        options = ["Choose an intent", *config.intents]
        intent = st.selectbox("True primary intent", options, index=options.index(row["intent"]) if row["intent"] in options else 0)
        decisions = ["Choose a decision", "auto_handle", "escalate"]
        decision = st.selectbox("Expected safe action", decisions, index=decisions.index(row["decision"]) if row["decision"] in decisions else 0)
        reason = st.text_input("Escalation reason (required for escalation)", value=row.get("escalation_reason", ""))
        points = st.text_area("Required reply points", value=row.get("required_reply_points", ""))
        forbidden = st.text_area("Forbidden claims (write none if truly none)", value=row.get("forbidden_claims", ""))
        confirmed = st.checkbox("I personally reviewed this example and these labels.")
        if st.form_submit_button("Save human annotation"):
            try:
                if not confirmed:
                    raise ValueError("Confirm your personal review before saving.")
                save_annotation(path, example_id, row["text"], intent, decision, reason,
                                points, forbidden, annotator, config.intents)
                st.success("Saved with backup. Select another example or refresh the page to update progress.")
            except (ValueError, OSError) as error:
                st.error(str(error))
else:
    path = config.data["human_scores_csv"]
    if not path.exists():
        st.info("First run evaluation, then judge, then prepare-human-scores. No scores have been invented or prefilled.")
        st.stop()
    worksheet = read_table(path)
    if worksheet.empty or not {"text", "draft_reply", "evidence_json", "cache_key", "reviewed"} <= set(worksheet.columns):
        st.info("First run evaluation, then judge, then prepare-human-scores. No scores have been invented or prefilled.")
        st.stop()
    st.caption("1 = poor, 3 = mixed, 5 = strong. Assess relevance, actionable next steps, evidence grounding, tone and safety independently.")
    include_done = st.checkbox("Include already scored replies")
    rows = worksheet if include_done else worksheet[~worksheet["reviewed"].eq("yes")]
    if rows.empty:
        st.success("No pending replies in this view.")
        st.stop()
    # Opaque choices omit system identity from the UI; source CSV is not fully blinded.
    index = st.selectbox("Reply item", rows.index.tolist(), format_func=lambda value: f"Item {value + 1}")
    row = rows.loc[index]
    st.subheader("Customer message")
    st.text(row["text"])
    st.subheader("Draft to score")
    st.text(row["draft_reply"])
    with st.expander("Historical evidence"):
        st.json(json.loads(row["evidence_json"]))
    with st.form(f"score-{row['example_id']}-{row['system']}"):
        scores = {}
        for field in RUBRIC_FIELDS:
            choices = ["Choose", 1, 2, 3, 4, 5]
            previous = int(float(row[field])) if row[field] else "Choose"
            scores[field] = st.selectbox(field.title(), choices, index=choices.index(previous))
        confirmed = st.checkbox("I scored this draft independently, without looking at judge scores.")
        if st.form_submit_button("Save human scores"):
            try:
                if not confirmed:
                    raise ValueError("Confirm independent scoring before saving.")
                save_human_score(path, row["example_id"], row["system"], row["cache_key"], scores, annotator)
                st.success("Saved with backup. Select another reply or refresh to update progress.")
            except (ValueError, OSError) as error:
                st.error(str(error))