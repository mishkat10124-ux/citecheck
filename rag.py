import glob
import os
import sys

import anthropic
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_WORDS, OVERLAP = 350, 60

_client = anthropic.Anthropic() # reads ANTHROPIC_API_KEY
_embedder = None


def embedder():
global _embedder
if _embedder is None:
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
return _embedder


def collection():
db = chromadb.PersistentClient(path=".chroma")
return db.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})


def chunk(text):
words = text.split()
step = CHUNK_WORDS - OVERLAP
return [
" ".join(words[i : i + CHUNK_WORDS])
for i in range(0, max(len(words) - OVERLAP, 1), step)
]


def ingest(folder="papers"):
col = collection()
for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
name = os.path.basename(path)
if col.get(ids=[f"{name}-0"])["ids"]:
print(f"skip (already ingested): {name}")
continue
text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
chunks = chunk(text)
if not chunks:
print(f"no text found: {name}")
continue
col.upsert(
ids=[f"{name}-{i}" for i in range(len(chunks))],
documents=chunks,
embeddings=embedder().encode(chunks).tolist(),
metadatas=[{"source": name, "chunk": i} for i in range(len(chunks))],
)
print(f"ingested {name}: {len(chunks)} chunks")


def retrieve(question, k=5):
res = collection().query(
query_embeddings=embedder().encode([question]).tolist(), n_results=k
)
return [
{"text": d, "source": m["source"]}
for d, m in zip(res["documents"][0], res["metadatas"][0])
]


def ask_claude(system, user, max_tokens=1000):
msg = _client.messages.create(
model=MODEL,
max_tokens=max_tokens,
system=system,
messages=[{"role": "user", "content": user}],
)
return "".join(b.text for b in msg.content if b.type == "text")


ANSWER_SYSTEM = (
"You answer questions about research papers using ONLY the numbered context "
"passages provided. Cite passages like [1], [2]. If the context does not "
"contain the answer, say you can't find it in the papers. Never use outside knowledge."
)


def answer(question, k=5):
ctx = retrieve(question, k)
numbered = "\n\n".join(f"[{i + 1}] ({c['source']}) {c['text']}" for i, c in enumerate(ctx))
reply = ask_claude(ANSWER_SYSTEM, f"Context:\n{numbered}\n\nQuestion: {question}")
return {"question": question, "answer": reply, "contexts": ctx}


if __name__ == "__main__":
if len(sys.argv) > 1 and sys.argv[1] == "ingest":
ingest(sys.argv[2] if len(sys.argv) > 2 else "papers")
elif len(sys.argv) > 1:
out = answer(" ".join(sys.argv[1:]))
print(out["answer"])
else:
print("usage: python rag.py ingest [folder] | python rag.py <question>")