"""
Upload page — lets the user upload a CSV and kicks off the graph run.

Build notes:
    - st.file_uploader for the CSV
    - on upload, read into a pandas DataFrame, build the initial
      AgentLabState, and call the compiled graph from
      graph/workflow.py
    - store the final state in st.session_state so other pages can
      read from it without re-running the pipeline
"""
import streamlit as st

st.title("Upload Dataset")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded:
    st.warning("Pipeline not wired in yet, this just previews the file for now.")
    import pandas as pd
    df = pd.read_csv(uploaded)
    st.write(f"{len(df)} rows, {len(df.columns)} columns")
    st.dataframe(df.head())
