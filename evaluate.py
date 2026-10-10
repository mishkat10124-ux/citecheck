import argparse
import glob
import json
import os
import random
import re
import sys
import time

from rag import answer, ask_claude, collection

METRICS = ["faithfulness", "answer_relevancy", "context_recall"]
REGRESSION_TOLERANCE = 0.05


def parse_json(text):
return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def make_testset(n, source=None):
col = collection()
data = col.get(where={"source": source}) if source else col.get()
pool = [d for d in data["documents"] if len(d) > 400]
random.seed(42)
items = []
for passage in random.sample(pool, min(n, len(pool))):
out = parse_json(
ask_claude(
"You write evaluation questions. Reply with JSON only.",
"Write one specific question answerable ONLY from this passage, plus a "
'short reference answer. Format: {"question": "...", "reference": "..."}'
f"\n\nPassage:\n{passage}",
)
)
items.append(out)
return items


def claim_support(text, contexts):
"""Fraction of claims in `text` that are supported by the contexts."""
ctx = "\n\n".join(c["text"] for c in contexts)
out = parse_json(
ask_claude(
"You are a strict fact-checker. Reply with JSON only.",
"Break the TEXT into atomic factual claims. For each, say whether it is "
"fully supported by the CONTEXT. Ignore citation markers like [1]. "
'Format: {"claims": [{"claim": "...", "supported": true}]}'
f"\n\nCONTEXT:\n{ctx}\n\nTEXT:\n{text}",
max_tokens=1500,
)
)
claims = out.get("claims", [])
if not claims:
return 1.0
return sum(bool(c["supported"]) for c in claims) / len(claims)


def relevancy(question, ans):
out = parse_json(
ask_claude(
"You grade answers. Reply with JSON only.",
"Rate 1-5 how directly and completely the ANSWER addresses the QUESTION "
"(1 = off-topic or refusal, 5 = fully addresses it). "
'Format: {"score": 4}'
f"\n\nQUESTION: {question}\n\nANSWER: {ans}",
)
)
return (out["score"] - 1) / 4


def previous_run():
files = sorted(glob.glob("runs/*.json"))
return json.load(open(files[-1])) if files else None


def main():
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=15, help="number of test questions")
ap.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
ap.add_argument("--source", help="only build questions from this PDF (e.g. a new paper)")
args = ap.parse_args()

rows = []
for item in make_testset(args.n, args.source):
res = answer(item["question"], args.k)
rows.append(
{
"question": item["question"],
"reference": item["reference"],
"answer": res["answer"],
"faithfulness": claim_support(res["answer"], res["contexts"]),
"answer_relevancy": relevancy(item["question"], res["answer"]),
"context_recall": claim_support(item["reference"], res["contexts"]),
}
)
print(f"{len(rows)}/{args.n} done")

means = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
prev = previous_run()

os.makedirs("runs", exist_ok=True)
path = f"runs/{time.strftime('%Y%m%d-%H%M%S')}.json"
json.dump({"config": vars(args), "means": means, "rows": rows}, open(path, "w"), indent=2)

print("\nMetric score vs last run")
regressed = False
for m in METRICS:
delta = "" if not prev else f"{means[m] - prev['means'][m]:+.2f}"
if prev and means[m] < prev["means"][m] - REGRESSION_TOLERANCE:
regressed = True
print(f"{m:<18} {means[m]:.2f} {delta}")
print(f"\nSaved {path}")
if regressed:
print("REGRESSION: a metric dropped more than the tolerance.")
sys.exit(1)


if __name__ == "__main__":
main()