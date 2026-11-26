"""Answer a question from retrieved statute passages, in points that each cite one passage
and restate it in a form that can be checked.

A first version asked for cited sentences and checked each with NLI against the cited
text. Almost nothing passed (median entailment 0.01): the model mixes the rule with its
application to the user's story ("Die Vereinbarung kann anfechtbar sein, wenn die Mieter
arglistig getäuscht haben ..."), and a statute never entails a claim about the user's
tenants. Checking was the right idea; the unit was wrong.

So every point of an answer has two parts:

- ``rule``: what the cited passage says, restated in plain German. This is checked: the
  passage text must entail it (multilingual NLI via belegt), or the point is dropped.
- ``relevance``: why the rule matters for this question. This is the model's reading of
  the case. It is shown, but labelled as an assessment, never as the statute.

The JSON schema only allows citing passages the model was shown, so a citation can't be
invented. If no point survives, the answer says the statutes found don't settle the
question.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from fundstelle.corpus import Passage
from fundstelle.llm import ChatBackend

SYSTEM = """Du erklärst Laien auf Deutsch, welche Gesetzesstellen für ihre Frage wichtig sind.
Benutze ausschließlich die unten aufgeführten Gesetzesstellen.
Gib höchstens vier Punkte zurück, die wichtigsten zuerst. Jeder Punkt hat:
- "cite": genau eine der aufgeführten Gesetzesstellen,
- "rule": was diese Gesetzesstelle regelt, in einfacher Sprache, eng am Wortlaut, in einem
  Satz, ohne Bezug auf den Fall der fragenden Person,
- "relevance": in einem Satz, warum diese Regel für die Frage wichtig ist.
Erfinde keine Fristen, Beträge oder Bedingungen. Wenn keine Gesetzesstelle zur Frage passt,
gib keine Punkte zurück."""

FALLBACK = (
    "Die gefundenen Gesetzesstellen beantworten Ihre Frage nicht sicher. Bitte wenden Sie "
    "sich an eine Rechtsberatung, z. B. eine Verbraucherzentrale oder einen Mieterverein."
)

Support = Callable[[str, str], float]
"""(premise, hypothesis) -> probability that the premise entails the hypothesis."""

THRESHOLD = 0.5


@dataclass
class Point:
    cite: str  # passage id, "BGB § 573 Abs. 1"
    rule: str
    relevance: str
    support: float | None = None  # NLI: does the cited passage entail ``rule``?

    @property
    def supported(self) -> bool:
        return self.support is not None and self.support >= THRESHOLD


@dataclass
class Answer:
    question: str
    passages: list[str]  # ids shown to the model
    draft: list[Point]  # what the model wrote
    kept: list[Point] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def abstained(self) -> bool:
        return not self.kept

    def text(self) -> str:
        if self.abstained:
            return FALLBACK
        return "\n".join(f"{cite(p.cite)}: {p.rule} Einschätzung: {p.relevance}" for p in self.kept)

    def cited_norms(self, kept: bool = True) -> list[str]:
        points = self.kept if kept else self.draft
        return list(dict.fromkeys(norm_of(p.cite) for p in points))


def cite(passage_id: str) -> str:
    """ "BGB § 573 Abs. 2" -> "§ 573 Abs. 2 BGB", the way it is written in German."""
    law, rest = passage_id.split(" ", 1)
    return f"{rest} {law}"


def norm_of(passage_id: str) -> str:
    return passage_id.split(" Abs. ")[0]


def schema(ids: Sequence[str]) -> dict:
    point = {
        "type": "object",
        "properties": {
            "cite": {"type": "string", "enum": list(ids)},
            "rule": {"type": "string"},
            "relevance": {"type": "string"},
        },
        "required": ["cite", "rule", "relevance"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"points": {"type": "array", "maxItems": 4, "items": point}},
        "required": ["points"],
        "additionalProperties": False,
    }


def prompt(question: str, passages: Sequence[Passage]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in passages)
    user = f"Gesetzesstellen:\n\n{context}\n\nFrage:\n{question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_BRACKETED = re.compile(r"\s*\[[^\]]*\]")


class Answerer:
    def __init__(self, llm: ChatBackend, support: Support | None = None) -> None:
        self.llm, self.support = llm, support

    def __call__(self, question: str, passages: Sequence[Passage]) -> Answer:
        by_id = {p.id: p for p in passages}
        out = self.llm.complete(prompt(question, passages), schema(list(by_id)))
        try:
            raw = json.loads(out.text)["points"]
        except (json.JSONDecodeError, KeyError, TypeError):
            raw = []
        draft, seen = [], set()
        for r in raw:
            rule = _BRACKETED.sub("", str(r.get("rule", ""))).strip()
            key = (r.get("cite"), rule)
            if r.get("cite") in by_id and rule and key not in seen:  # small models repeat
                seen.add(key)
                draft.append(Point(r["cite"], rule, str(r.get("relevance", "")).strip()))
        answer = Answer(question, list(by_id), draft, seconds=out.seconds)
        if self.support:
            for p in draft:
                passage = by_id[p.cite]
                p.support = self.support(f"{passage.title}: {passage.text}", p.rule)
        answer.kept = [p for p in draft if self.support is None or p.supported]
        return answer


class NLISupport:
    """``Support`` backed by belegt's multilingual NLI model (loaded on first use)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name, self._model = model_name, None

    def __call__(self, premise: str, hypothesis: str) -> float:
        if self._model is None:
            from belegt.nli import DEFAULT_MODEL, NLIModel

            self._model = NLIModel(self.model_name or DEFAULT_MODEL)
        (p,) = self._model.predict([(premise, hypothesis)])
        return p["entailment"]
