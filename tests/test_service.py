from fastapi.testclient import TestClient

from fundstelle import service
from fundstelle.answer import Answerer
from fundstelle.llm import ScriptedBackend
from fundstelle.retrieval import BM25
from tests.test_answer import P, fake_support, point


def client(responses):
    service.STATE.update(passages=P, retriever=BM25(P),
                         answerer=Answerer(ScriptedBackend(responses), fake_support))  # fmt: skip
    return TestClient(service.app)  # no `with`: the lifespan (real models) is not run


def test_search_and_ask_return_cited_and_checked_points():
    c = client([{"points": [
        point("BGB § 573 Abs. 1", "Er braucht ein berechtigtes Interesse."),
        point("BGB § 573 Abs. 1", "Die Frist beträgt drei Monate."),
    ]}])  # fmt: skip
    hits = c.post("/search", json={"question": "Darf mein Vermieter kündigen?", "k": 2}).json()
    assert hits["passages"][0]["cite"] == "§ 573 Abs. 1 BGB"
    r = c.post("/ask", json={"question": "Darf mein Vermieter kündigen?"}).json()
    assert r["answer"].startswith("§ 573 Abs. 1 BGB: Er braucht ein berechtigtes Interesse.")
    assert [p["kept"] for p in r["points"]] == [True, False]
    assert r["trace"]["drafted"] == 2 and not r["abstained"]
    assert 'fundstelle_requests_total{kind="ask"}' in c.get("/metrics").text


def test_short_questions_are_rejected():
    assert client([]).post("/ask", json={"question": "kurz"}).status_code == 422
