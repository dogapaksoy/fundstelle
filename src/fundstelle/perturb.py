"""How many wrong rules would the NLI check let through?

The answers evaluation shows how often the check *rejects* a rule, not how often it
should have. To measure that, take the rules it accepted and break them the way a model
gets statutes wrong, then check again:

- **negated**: the rule says the opposite ("kann" -> "kann nicht", "nicht" removed)
- **number**: a number or number word changes ("zwei Wochen" -> "drei Wochen")
- **wrong passage**: the unchanged rule is checked against another passage that was shown
  for the same question (a correct sentence with the wrong citation)

A rejected perturbation is a caught error.
"""

from __future__ import annotations

import random
import re

from fundstelle.answer import THRESHOLD, Support
from fundstelle.corpus import Passage

_FLIP = [  # first match wins
    (r"\bnur\b", "nicht"),
    (r"\bkann nicht\b", "kann"), (r"\bkann\b(?! nur)", "kann nicht"),
    (r"\bmuss nicht\b", "muss"), (r"\bmuss\b(?! nur)", "muss nicht"),
    (r"\bdarf nicht\b", "darf"), (r"\bdarf\b(?! nur)", "darf nicht"),
    (r"\bist nicht\b", "ist"), (r"\bist\b(?! nur)", "ist nicht"),
    (r"\bhat nicht\b", "hat"), (r"\bhat\b(?! nur)", "hat nicht"),
    (r"\bkeine\b", "eine"), (r"\bkein\b", "ein"),
]  # fmt: skip
_WORDS = ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
          "elf", "zwölf"]  # fmt: skip
_NUMBER = re.compile(rf"\b(\d+|{'|'.join(_WORDS)})\b")


def negate(rule: str) -> str | None:
    for pattern, repl in _FLIP:
        new, n = re.subn(pattern, repl, rule, count=1)
        if n:
            return new
    return None


def change_number(rule: str) -> str | None:
    m = _NUMBER.search(rule)
    if not m:
        return None
    x = m.group(1)
    new = str(int(x) * 2 + 1) if x.isdigit() else _WORDS[(_WORDS.index(x) + 2) % len(_WORDS)]
    return rule[: m.start()] + new + rule[m.end() :]


def run(rows: list[dict], by_id: dict[str, Passage], support: Support, seed: int = 0) -> dict:
    """``rows`` from ``evaluate_answers``; returns per-kind counts of (tested, rejected)."""
    rng = random.Random(seed)
    out: dict[str, list[int]] = {"original": [0, 0], "negated": [0, 0], "number": [0, 0],
                                 "wrong passage": [0, 0]}  # fmt: skip
    examples: list[dict] = []

    def check(kind: str, passage: Passage, rule: str) -> None:
        s = support(f"{passage.title}: {passage.text}", rule)
        out[kind][0] += 1
        out[kind][1] += s < THRESHOLD
        if len(examples) < 40:
            examples.append({"kind": kind, "cite": passage.id, "rule": rule, "support": s})

    for r in rows:
        shown = [by_id[c] for c in dict.fromkeys(p["cite"] for p in r["draft"])]
        for p in r["kept"]:
            passage = by_id[p["cite"]]
            out["original"][0] += 1
            for kind, new in (("negated", negate(p["rule"])), ("number", change_number(p["rule"]))):
                if new:
                    check(kind, passage, new)
            others = [x for x in shown if x.norm_id != passage.norm_id]
            if others:
                check("wrong passage", rng.choice(others), p["rule"])
    out["original"][1] = 0  # kept rules passed by definition
    return {"counts": out, "examples": examples}
