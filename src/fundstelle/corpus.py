"""German federal statutes as retrievable passages.

The official XML from gesetze-im-internet.de is split into one passage per Absatz
("BGB § 573 Abs. 2"), because that is the unit lawyers cite and the unit an answer
sentence can be checked against. Each passage keeps its norm's title, so "(4) Eine zum
Nachteil des Mieters abweichende Vereinbarung ist unwirksam." is still findable as part
of "Ordentliche Kündigung des Vermieters".
"""

from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

# law abbreviation as cited -> path on gesetze-im-internet.de
LAWS = {"BGB": "bgb", "StGB": "stgb", "ZPO": "zpo"}
SOURCE = "https://www.gesetze-im-internet.de/{}/xml.zip"

_ABSATZ = re.compile(r"^\((\d+[a-z]?)\)\s*")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Passage:
    id: str  # "BGB § 573 Abs. 2"
    law: str  # "BGB"
    norm: str  # "§ 573"
    title: str  # "Ordentliche Kündigung des Vermieters"
    text: str

    @property
    def norm_id(self) -> str:
        return f"{self.law} {self.norm}"

    def for_index(self) -> str:
        return f"{self.norm_id} {self.title}: {self.text}"


def _text(el: ET.Element) -> str:
    """Flatten a <P> element. List items keep their numbers ("1. der Mieter ...")."""
    for dt in el.iter("DT"):
        dt.text = (dt.text or "") + " "
    return _SPACE.sub(" ", "".join(el.itertext())).strip()


def parse(xml: bytes | str, law: str) -> Iterator[Passage]:
    root = ET.fromstring(xml)
    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        enbez = meta.findtext("enbez") if meta is not None else None
        if not enbez or not enbez.startswith("§"):
            continue  # headings, tables of contents, annexes
        enbez = _SPACE.sub(" ", enbez).strip()
        title = _SPACE.sub(" ", meta.findtext("titel") or "").strip()
        content = norm.find("textdaten/text/Content")
        if content is None:
            continue
        paras = [_text(p) for p in content.findall("P")]
        paras = [p for p in paras if p]
        if not paras or (len(paras) == 1 and paras[0] in {"-", "(weggefallen)"}):
            continue
        if title == "(weggefallen)":
            continue
        numbered = [(_ABSATZ.match(p), p) for p in paras]
        if all(m for m, _ in numbered):
            for m, p in numbered:
                yield Passage(f"{law} {enbez} Abs. {m.group(1)}", law, enbez, title, p)
        else:  # a norm without Absätze is one passage
            yield Passage(f"{law} {enbez}", law, enbez, title, " ".join(paras))


def download(data: Path, laws: Iterable[str] = LAWS) -> Path:
    """Fetch the XML of each law and write all passages to ``data/corpus.jsonl``."""
    import urllib.request

    data.mkdir(parents=True, exist_ok=True)
    out = data / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for law in laws:
            raw = urllib.request.urlopen(SOURCE.format(LAWS[law]), timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                (name,) = [n for n in z.namelist() if n.endswith(".xml")]
                for p in parse(z.read(name), law):
                    f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return out


def norm_docs(passages: Iterable[Passage]) -> list[Passage]:
    """One document per norm (all its Absätze), for retrieving norms rather than Absätze."""
    grouped: dict[str, list[Passage]] = {}
    for p in passages:
        grouped.setdefault(p.norm_id, []).append(p)
    return [
        Passage(nid, ps[0].law, ps[0].norm, ps[0].title, " ".join(p.text for p in ps))
        for nid, ps in grouped.items()
    ]


def load(path: Path) -> list[Passage]:
    return [Passage(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines()]
