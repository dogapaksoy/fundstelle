from fundstelle.answer import FALLBACK, Answerer, cite, schema
from fundstelle.corpus import Passage
from fundstelle.llm import ScriptedBackend

P = [
    Passage(
        "BGB § 573 Abs. 1",
        "BGB",
        "§ 573",
        "Ordentliche Kündigung des Vermieters",
        "(1) Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse hat.",
    ),
    Passage("BGB § 1", "BGB", "§ 1", "Beginn der Rechtsfähigkeit", "Die Rechtsfähigkeit ..."),
]


def fake_support(premise: str, hypothesis: str) -> float:
    return 0.9 if "berechtigtes Interesse" in premise and "Interesse" in hypothesis else 0.1


def test_unsupported_sentences_are_dropped_and_citations_formatted():
    llm = ScriptedBackend([{"sentences": [
        {"text": "Der Vermieter braucht ein berechtigtes Interesse.",
         "cites": ["BGB § 573 Abs. 1"]},
        {"text": "Die Frist beträgt drei Monate.", "cites": ["BGB § 573 Abs. 1"]},
    ]}])  # fmt: skip
    a = Answerer(llm, fake_support)("Darf mein Vermieter kündigen?", P)
    assert [s.text for s in a.kept] == ["Der Vermieter braucht ein berechtigtes Interesse."]
    assert a.text() == "Der Vermieter braucht ein berechtigtes Interesse. (§ 573 Abs. 1 BGB)"
    assert a.cited_norms() == ["BGB § 573"]
    assert len(a.draft) == 2


def test_no_sentence_left_means_fallback():
    llm = ScriptedBackend([{"sentences": [{"text": "Die Frist ist kurz.", "cites": ["BGB § 1"]}]}])
    a = Answerer(llm, fake_support)("Wie lange ist die Frist?", P)
    assert a.abstained and a.text() == FALLBACK


def test_schema_only_allows_shown_citations_and_bad_json_abstains():
    s = schema(["BGB § 1"])
    assert s["properties"]["sentences"]["items"]["properties"]["cites"]["items"]["enum"] == [
        "BGB § 1"
    ]
    assert Answerer(ScriptedBackend(["not json"]))("?", P).abstained
    assert cite("StGB § 242 Abs. 2") == "§ 242 Abs. 2 StGB"
