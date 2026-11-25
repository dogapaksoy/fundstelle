"""Retrieval metrics against the paragraphs lawyers cited.

A question counts as a hit@k if any norm the lawyer cited is among the first k distinct
norms retrieved: that is what the answer generator gets to read. Recall@k is the share
of all cited norms found, MRR the reciprocal rank of the first cited norm.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fundstelle.corpus import Passage
from fundstelle.data import Question
from fundstelle.retrieval import Ranking, norms


@dataclass
class Scores:
    name: str
    n: int
    hit: dict[int, float]
    recall: dict[int, float]
    mrr: float

    def row(self, ks: Sequence[int] = (1, 5, 10, 20)) -> str:
        hits = " | ".join(f"{self.hit[k]:.1%}" for k in ks)
        return f"| {self.name} | {hits} | {self.recall[10]:.1%} | {self.mrr:.3f} |"


def header(ks: Sequence[int] = (1, 5, 10, 20), timing: bool = False) -> str:
    cols = " | ".join(f"hit@{k}" for k in ks) + " | recall@10 | MRR"
    n = len(ks) + 2
    if timing:
        cols, n = cols + " | s / question", n + 1
    return f"| retriever | {cols} |\n|---|{'---|' * n}"


def per_question(q: Question, ranked: list[str]) -> dict:
    gold = set(q.gold)
    first = next((r for r, n in enumerate(ranked, 1) if n in gold), None)
    return {"id": q.id, "law": q.law, "gold": list(q.gold), "top": ranked[:20], "first": first}


def score(name: str, rows: list[dict], ks: Sequence[int] = (1, 5, 10, 20)) -> Scores:
    hit = {k: statistics.mean(r["first"] is not None and r["first"] <= k for r in rows) for k in ks}
    recall = {
        k: statistics.mean(len(set(r["gold"]) & set(r["top"][:k])) / len(r["gold"]) for r in rows)
        for k in ks
    }
    mrr = statistics.mean(1 / r["first"] if r["first"] else 0.0 for r in rows)
    return Scores(name, len(rows), hit, recall, mrr)


def run(name: str, rankings: list[Ranking], questions: list[Question],
        passages: Sequence[Passage], out: Path | None = None) -> list[dict]:  # fmt: skip
    rows = [per_question(q, norms(r, passages)) for q, r in zip(questions, rankings, strict=True)]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return rows
