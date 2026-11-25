"""GerLayQA: real questions by laypeople, answered by lawyers who cite the statute.

Büttner & Habernal (2024), "Answering legal questions from laymen in German civil law
system", EACL. Each record is a question from an online legal advice forum, the lawyer's
answer, and the paragraphs the answer cites ("§ 573"). The cited paragraphs are the gold
labels for retrieval: did the system find the norm a lawyer used?

Questions that already name one of their gold paragraphs (3–5%) are dropped from the
evaluation splits; for them retrieval is string matching.
"""

from __future__ import annotations

import html
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

REPO = "ViolaCamille/GerLayQA"
FILES = {"BGB": "GerLayQA.json", "StGB": "stgb_QA.json", "ZPO": "zpo_QA.json"}
# questions per law in each split; StGB and ZPO are small, so they are not proportional
SPLITS = {"dev": {"BGB": 120, "StGB": 40, "ZPO": 40}, "test": {"BGB": 600, "StGB": 200, "ZPO": 200}}


@dataclass(frozen=True)
class Question:
    id: str  # "BGB-1234": law and index in the source file
    law: str
    text: str
    answer: str
    gold: tuple[str, ...]  # ("BGB § 573", "BGB § 573c")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def names_gold(q: Question) -> bool:
    return any(re.search(re.escape(g.split(" ", 1)[1]) + r"(?![\da-z])", q.text) for g in q.gold)


def load_raw(data: Path, law: str) -> list[Question]:
    path = data / "gerlayqa" / FILES[law]
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO, FILES[law], repo_type="dataset", local_dir=data / "gerlayqa")
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, r in enumerate(rows):
        gold = tuple(dict.fromkeys(f"{law} {p}" for p in r["Paragraphs"]))
        out.append(
            Question(f"{law}-{i}", law, clean(r["Question_text"]), clean(r["Answer_text"]), gold)
        )
    return out


def splits(data: Path, norms: set[str], seed: int = 0) -> dict[str, list[Question]]:
    """Disjoint dev and test samples. Only questions whose gold norms all exist in the
    current statutes (99%: a few cite repealed paragraphs) and that don't name them."""
    out: dict[str, list[Question]] = {name: [] for name in SPLITS}
    for law in FILES:
        seen: set[str] = set()
        pool = []
        for q in load_raw(data, law):
            if q.text in seen or not q.gold or names_gold(q):
                continue
            seen.add(q.text)
            if all(g in norms for g in q.gold):
                pool.append(q)
        random.Random(f"{seed}-{law}").shuffle(pool)
        start = 0
        for name, sizes in SPLITS.items():
            out[name] += pool[start : start + sizes[law]]
            start += sizes[law]
    return out
