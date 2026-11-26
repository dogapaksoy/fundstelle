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


def point(cite, rule, relevance="Das betrifft Ihren Fall."):
    return {"cite": cite, "rule": rule, "relevance": relevance}


def test_unsupported_rules_are_dropped_and_repeats_removed():
    rule = "Der Vermieter braucht ein berechtigtes Interesse [BGB § 573 Abs. 1]."
    llm = ScriptedBackend([{"points": [
        point("BGB § 573 Abs. 1", rule), point("BGB § 573 Abs. 1", rule),
        point("BGB § 573 Abs. 1", "Die Frist beträgt drei Monate."),
    ]}])  # fmt: skip
    a = Answerer(llm, fake_support)("Darf mein Vermieter kündigen?", P)
    assert len(a.draft) == 2
    assert [p.rule for p in a.kept] == ["Der Vermieter braucht ein berechtigtes Interesse."]
    assert a.text() == (
        "§ 573 Abs. 1 BGB: Der Vermieter braucht ein berechtigtes Interesse. "
        "Einschätzung: Das betrifft Ihren Fall."
    )
    assert a.cited_norms() == ["BGB § 573"]


def test_no_point_left_means_fallback():
    llm = ScriptedBackend([{"points": [point("BGB § 1", "Die Frist ist kurz.")]}])
    a = Answerer(llm, fake_support)("Wie lange ist die Frist?", P)
    assert a.abstained and a.text() == FALLBACK


def test_schema_only_allows_shown_citations_and_bad_json_abstains():
    s = schema(["BGB § 1"])
    assert s["properties"]["points"]["items"]["properties"]["cite"]["enum"] == ["BGB § 1"]
    assert Answerer(ScriptedBackend(["not json"]))("?", P).abstained
    assert cite("StGB § 242 Abs. 2") == "§ 242 Abs. 2 StGB"


def test_perturbations_flip_meaning_and_numbers():
    from fundstelle.perturb import change_number, negate, run

    assert negate("Der Vermieter kann nur kündigen, wenn er ein Interesse hat.") == (
        "Der Vermieter kann nicht kündigen, wenn er ein Interesse hat."
    )
    assert negate("Der Käufer muss nicht zahlen.") == "Der Käufer muss zahlen."
    assert change_number("Die Frist beträgt zwei Wochen.") == "Die Frist beträgt vier Wochen."
    kept = {"cite": "BGB § 573 Abs. 1", "rule": "Er muss ein berechtigtes Interesse haben."}
    row = {"draft": [{"cite": p.id} for p in P], "kept": [kept]}
    counts = run([row], {p.id: p for p in P}, fake_support)["counts"]
    assert counts["wrong passage"] == [1, 1]  # § 1 does not support it
