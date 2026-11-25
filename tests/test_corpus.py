from fundstelle.corpus import parse
from fundstelle.data import Question, names_gold
from fundstelle.retrieval import BM25, Hybrid, norms

XML = """<dokumente>
<norm><metadaten><jurabk>BGB</jurabk><gliederungseinheit/></metadaten></norm>
<norm><metadaten><jurabk>BGB</jurabk><enbez>§ 573</enbez>
<titel format="parat">Ordentliche Kündigung des Vermieters</titel></metadaten>
<textdaten><text format="XML"><Content>
<P>(1) Der Vermieter kann nur kündigen, wenn er ein berechtigtes Interesse hat.</P>
<P>(2) Ein berechtigtes Interesse liegt vor, wenn <DL><DT>1.</DT><DD><LA>der Mieter seine
Pflichten verletzt hat,</LA></DD></DL></P>
</Content></text></textdaten></norm>
<norm><metadaten><jurabk>BGB</jurabk><enbez>§ 1</enbez>
<titel format="parat">Beginn der Rechtsfähigkeit</titel></metadaten>
<textdaten><text format="XML"><Content><P>Die Rechtsfähigkeit des Menschen beginnt mit der
Vollendung der Geburt.</P></Content></text></textdaten></norm>
<norm><metadaten><jurabk>BGB</jurabk><enbez>§ 2a</enbez><titel>(weggefallen)</titel></metadaten>
<textdaten><text format="XML"><Content><P>-</P></Content></text></textdaten></norm>
</dokumente>"""


def test_parse_splits_absaetze_and_skips_headings():
    ps = list(parse(XML, "BGB"))
    assert [p.id for p in ps] == ["BGB § 573 Abs. 1", "BGB § 573 Abs. 2", "BGB § 1"]
    assert ps[1].text == (
        "(2) Ein berechtigtes Interesse liegt vor, wenn 1. der Mieter seine Pflichten verletzt hat,"
    )
    assert ps[0].norm_id == "BGB § 573"
    assert ps[0].title == "Ordentliche Kündigung des Vermieters"


def test_names_gold_respects_suffixes():
    q = Question("x", "BGB", "Gilt § 573a hier?", "", ("BGB § 573",))
    assert not names_gold(q)
    assert names_gold(Question("x", "BGB", "Gilt § 573 hier?", "", ("BGB § 573",)))


def test_bm25_and_fusion_rank_the_matching_norm_first():
    ps = list(parse(XML, "BGB"))
    bm = BM25(ps)
    (ranking,) = bm.search(["Darf mein Vermieter mir kündigen?"], 3)
    assert norms(ranking, ps)[0] == "BGB § 573"
    (fused,) = Hybrid([bm, bm]).search(["Wann beginnt die Rechtsfähigkeit?"], 3)
    assert norms(fused, ps)[0] == "BGB § 1"
