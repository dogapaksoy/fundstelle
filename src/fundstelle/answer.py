"""Answer a question from retrieved statute passages; every sentence must cite and be
supported by them.

1. The model sees the question and the top passages, labelled with their citation
   ("BGB § 573 Abs. 2"). It answers in at most five sentences, each with the passages it
   rests on. The JSON schema only allows the labels it was shown, so a citation can't
   be invented, only misapplied.
2. Each sentence is checked against the text of the passages it cites with a multilingual
   NLI model (via belegt). A sentence the cited text does not entail is dropped.
3. If nothing survives, the answer says the statutes found don't settle the question,
   instead of guessing.

The check is deliberately strict: a sentence that applies a rule to the user's facts
("Ihr Vermieter darf also nicht kündigen") is often not entailed by the statute alone.
The trade-off is measured in ``evaluate_answers``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from fundstelle.corpus import Passage
from fundstelle.llm import ChatBackend

SYSTEM = """Du beantwortest Rechtsfragen von Laien auf Deutsch, in einfacher Sprache.
Benutze ausschließlich die unten aufgeführten Gesetzesstellen. Jeder Satz deiner Antwort
muss die Gesetzesstellen angeben, auf denen er beruht. Gib die Regel aus dem Gesetz
wieder; erfinde keine Fristen, Beträge oder Bedingungen, die nicht im Text stehen.
Wenn die Gesetzesstellen die Frage nicht beantworten, gib keine Sätze zurück.
Antworte in höchstens fünf Sätzen."""

FALLBACK = (
    "Die gefundenen Gesetzesstellen beantworten Ihre Frage nicht sicher. Bitte wenden Sie "
    "sich an eine Rechtsberatung, z. B. eine Verbraucherzentrale oder einen Mieterverein."
)

Support = Callable[[str, str], float]
"""(premise, hypothesis) -> probability that the premise entails the hypothesis."""


@dataclass
class Sentence:
    text: str
    cites: list[str]
    support: float | None = None  # NLI entailment by the cited passages

    @property
    def supported(self) -> bool:
        return self.support is not None and self.support >= 0.5


@dataclass
class Answer:
    question: str
    passages: list[str]  # ids shown to the model
    draft: list[Sentence]  # what the model wrote
    kept: list[Sentence] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def abstained(self) -> bool:
        return not self.kept

    def text(self) -> str:
        if self.abstained:
            return FALLBACK
        return " ".join(f"{s.text} ({'; '.join(map(cite, s.cites))})" for s in self.kept)

    def cited_norms(self, kept: bool = True) -> list[str]:
        sents = self.kept if kept else self.draft
        return list(dict.fromkeys(norm_of(c) for s in sents for c in s.cites))


def cite(passage_id: str) -> str:
    """ "BGB § 573 Abs. 2" -> "§ 573 Abs. 2 BGB", the way it is written in German."""
    law, rest = passage_id.split(" ", 1)
    return f"{rest} {law}"


def norm_of(passage_id: str) -> str:
    return passage_id.split(" Abs. ")[0]


def schema(ids: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "cites": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {"type": "string", "enum": list(ids)},
                        },
                    },
                    "required": ["text", "cites"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }


def prompt(question: str, passages: Sequence[Passage]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in passages)
    user = f"Gesetzesstellen:\n\n{context}\n\nFrage:\n{question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class Answerer:
    def __init__(self, llm: ChatBackend, support: Support | None = None) -> None:
        self.llm, self.support = llm, support

    def __call__(self, question: str, passages: Sequence[Passage]) -> Answer:
        by_id = {p.id: p for p in passages}
        out = self.llm.complete(prompt(question, passages), schema(list(by_id)))
        try:
            raw = json.loads(out.text)["sentences"]
        except (json.JSONDecodeError, KeyError, TypeError):
            raw = []
        draft = [
            Sentence(s["text"].strip(), [c for c in dict.fromkeys(s["cites"]) if c in by_id])
            for s in raw
            if s.get("text", "").strip()
        ]
        answer = Answer(question, list(by_id), draft, seconds=out.seconds)
        for s in draft:
            if s.cites and self.support:
                premise = " ".join(f"{by_id[c].title}: {by_id[c].text}" for c in s.cites)
                s.support = self.support(premise, s.text)
        answer.kept = [s for s in draft if s.cites and (self.support is None or s.supported)]
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
