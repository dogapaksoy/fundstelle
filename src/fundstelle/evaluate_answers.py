"""End-to-end answers on GerLayQA questions.

For each question the answerer sees the top passages of a retriever and writes a cited
answer. Measured per question:

- **cites a lawyer's norm**: the answer cites at least one norm the lawyer cited. This
  is a lower bound on usefulness (lawyers cite a few norms, not every relevant one).
- **sentences supported**: share of drafted sentences the cited passages entail (NLI).
- **abstained**: nothing survived the check, the user gets the fallback.

``oracle`` mode puts the passages of the lawyer's norms first. It separates retrieval
errors (the right norm was never shown) from generation errors.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from fundstelle.answer import Answerer
from fundstelle.corpus import Passage
from fundstelle.data import Question
from fundstelle.retrieval import Ranking


def context(q: Question, ranking: Ranking, passages: Sequence[Passage], k: int,
            oracle: bool = False) -> list[Passage]:  # fmt: skip
    ranked = [passages[i] for i, _ in ranking]
    if oracle:
        gold = [p for p in passages if p.norm_id in q.gold][:k]
        ranked = gold + [p for p in ranked if p.norm_id not in q.gold]
    return ranked[:k]


def run(
    name: str,
    answerer: Answerer,
    questions: list[Question],
    rankings: list[Ranking],
    passages: Sequence[Passage],
    out: Path,
    k: int = 8,
    oracle: bool = False,
) -> list[dict]:
    """Answers each question once; rows already in ``out`` are reused (resumable)."""
    done = {}
    if out.exists():
        done = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a", encoding="utf-8") as f:
        for q, ranking in zip(questions, rankings, strict=True):
            if q.id in done:
                rows.append(done[q.id])
                continue
            shown = context(q, ranking, passages, k, oracle)
            a = answerer(q.text, shown)
            row = {
                "id": q.id, "condition": name, "gold": list(q.gold),
                "shown_gold": any(p.norm_id in q.gold for p in shown),
                "draft": [asdict(s) for s in a.draft], "kept": [asdict(s) for s in a.kept],
                "cited_draft": a.cited_norms(kept=False), "cited": a.cited_norms(),
                "text": a.text(), "seconds": a.seconds,
            }  # fmt: skip
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(f"{len(rows)}/{len(questions)} {q.id} kept {len(a.kept)}/{len(a.draft)} "
                  f"{a.seconds:.0f}s", flush=True)  # fmt: skip
    return rows


def summary(name: str, rows: list[dict]) -> str:
    def pct(xs) -> str:
        xs = list(xs)
        return f"{sum(xs) / len(xs):.0%}" if xs else "–"

    drafted = [s for r in rows for s in r["draft"]]
    supported = [s["support"] is not None and s["support"] >= 0.5 for s in drafted]
    return " | ".join([
        f"| {name}", str(len(rows)), pct(r["shown_gold"] for r in rows),
        pct(bool(set(r["cited_draft"]) & set(r["gold"])) for r in rows),
        pct(bool(set(r["cited"]) & set(r["gold"])) for r in rows),
        pct(supported), pct(not r["kept"] for r in rows),
        f"{statistics.median(r['seconds'] for r in rows):.0f} |",
    ])  # fmt: skip


HEADER = (
    "| condition | questions | lawyer's norm shown | draft cites lawyer's norm | answer cites "
    "lawyer's norm | sentences supported | abstained | s / answer |\n"
    "|---|---|---|---|---|---|---|---|"
)
