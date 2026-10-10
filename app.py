import glob
import json

import streamlit as st

from rag import answer

st.title("Research Paper Q&A")
q = st.text_input("Ask a question about your papers")
k = st.sidebar.slider("Chunks retrieved (k)", 1, 10, 5)

if q:
res = answer(q, k)
st.write(res["answer"])
with st.expander("Sources"):
for i, c in enumerate(res["contexts"], 1):
st.markdown(f"**[{i}] {c['source']}**")
st.caption(c["text"][:500] + "...")

st.sidebar.header("Latest eval run")
runs = sorted(glob.glob("runs/*.json"))
if runs:
last = json.load(open(runs[-1]))
for metric, val in last["means"].items():
st.sidebar.metric(metric, f"{val:.2f}")
else:
st.sidebar.caption("No runs yet. Run: python evaluate.py")