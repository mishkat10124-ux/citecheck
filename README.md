# CiteCheck

Research paper Q&A with an automated evaluation loop that scores **faithfulness** and **relevancy**.

Most "chat with your PDF" projects stop at answering. CiteCheck also measures *how good the answers are*, so you can change a setting (chunk size, retrieval depth, prompt) and see whether quality went up or down.

## How it works

```
PDFs -> chunk -> embed -> Chroma -> retrieve top-k -> Claude answers with citations
|
eval loop: generate test questions -> score each answer
```

1. **Ingest:** PDFs in `papers/` are split into ~350-word chunks (60-word overlap), embedded with `all-MiniLM-L6-v2`, and stored in a local Chroma database.
2. **Answer:** the top-k chunks are retrieved and Claude answers using only that context, citing passages like [1], [2]. If the answer isn't in the papers, it says so.
3. **Evaluate:** `evaluate.py` samples passages, has Claude write a question and reference answer for each, runs the full pipeline, and scores the results.

## Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Share of claims in the answer that are supported by the retrieved text (catches hallucination) |
| Answer relevancy | How directly the answer addresses the question (1-5, normalized to 0-1) |
| Context recall | Share of the reference answer's claims found in the retrieved text (catches retrieval misses) |

Every run is saved to `runs/` and compared to the previous one. The script exits with an error if any metric drops by more than 0.05, so it can act as a regression check.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key # PowerShell: $env:ANTHROPIC_API_KEY="your_key"
mkdir papers # add PDFs (e.g. from arXiv)
python rag.py ingest
```

## Usage

```bash
# Ask a question
python rag.py "What method does the paper propose?"

# Web UI
streamlit run app.py

# Run the evaluation (15 generated questions, 5 chunks retrieved each)
python evaluate.py --n 15 --k 5

# Evaluate on a newly added paper only
python evaluate.py --source newpaper.pdf
```

## Results

Replace this table with your own numbers after running `evaluate.py`.

| Setting | Faithfulness | Answer relevancy | Context recall |
|---|---|---|---|
| Baseline (k=5, 350-word chunks) | _ | _ | _ |
| k=3 | _ | _ | _ |
| k=8 | _ | _ | _ |

## Limitations

- The same model family writes the questions, generates answers, and judges them, which can inflate scores. Using a different model as judge is a good next step.
- Questions are generated from single passages, so multi-paper or multi-hop questions aren't covered.
- Scanned PDFs without a text layer can't be ingested.

## Files

- `rag.py`: ingestion, retrieval, answering
- `evaluate.py`: test set generation and scoring
- `app.py`: Streamlit UI