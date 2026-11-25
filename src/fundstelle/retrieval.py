"""Retrievers over statute passages: lexical (BM25), dense (embeddings), and hybrid.

Lay questions and statutes use different words: "Vermieter will mich rauswerfen" vs.
"ordentliche Kündigung des Vermieters", "Nachbar" vs. "Eigentümer". BM25 finds exact
terms, embeddings find paraphrases; reciprocal rank fusion combines them without having
to calibrate their scores against each other.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from fundstelle.corpus import Passage

Ranking = list[tuple[int, float]]  # (passage index, score), best first


class Retriever(Protocol):
    name: str

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]: ...


# ---------------------------------------------------------------------------- lexical

_WORD = re.compile(r"[^\W\d_]+|\d+[a-z]?", re.UNICODE)
_STOPWORDS = (
    "aber alle allem allen aller alles als also am an ander andere anderem anderen anderer "
    "anderes auch auf aus bei bin bis bist da damit dann das dass dein deine dem den der "
    "des dich die dies diese diesem diesen dieser dieses dir doch dort du durch ein eine "
    "einem einen einer eines er es etwas euch euer für gegen hab habe haben hat hatte hätte "
    "ich ihm ihn ihnen ihr ihre im in ist ja jede jedem jeden jeder jedes jetzt kann kein "
    "keine können man mein meine mich mir mit muss nach nicht nichts noch nun nur ob oder "
    "ohne sehr sein seine sich sie sind so soll sollte um und uns unser unter vom von vor "
    "war waren was weil wenn wer werden wie wieder will wir wird wo zu zum zur über"
)
_STOP = set(_STOPWORDS.split())


class Tokenizer:
    def __init__(self) -> None:
        import snowballstemmer

        self._stem = snowballstemmer.stemmer("german").stemWord
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        out = []
        for w in _WORD.findall(text.lower()):
            if w in _STOP or len(w) < 2:
                continue
            if w not in self._cache:
                self._cache[w] = self._stem(w)
            out.append(self._cache[w])
        return out


class BM25:
    name = "bm25"

    def __init__(self, passages: Sequence[Passage], k1: float = 1.2, b: float = 0.75) -> None:
        self.tok = Tokenizer()
        docs = [self.tok(p.for_index()) for p in passages]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, docs)) / len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        df: Counter[str] = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                self.postings.setdefault(t, []).append((i, tf))
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
        for t in set(self.tok(query)):
            if t not in self.postings:
                continue
            idx, tf = zip(*self.postings[t], strict=True)
            idx_a, tf_a = np.array(idx), np.array(tf, dtype=np.float32)
            s[idx_a] += self.idf[t] * tf_a * (self.k1 + 1) / (tf_a + norm[idx_a])
        return s

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        return [_top(self.scores(q), k) for q in queries]


# ---------------------------------------------------------------------------- dense

# How each model wants queries and passages to be marked.
PREFIXES = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
}


class Dense:
    def __init__(
        self,
        passages: Sequence[Passage],
        model: str = "BAAI/bge-m3",
        cache: Path | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        prefixes: tuple[str, str] | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"dense:{model.split('/')[-1]}"
        self.model = SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = max_length
        self.q_prefix, p_prefix = prefixes or PREFIXES.get(model, ("", ""))
        self.batch_size = batch_size
        texts = [p_prefix + p.for_index() for p in passages]
        digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:10]
        path = cache / f"emb-{model.strip('/').replace('/', '_')}-{digest}.npy" if cache else None
        if path and path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.encode(texts, show_progress=True)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.emb)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        ).astype(np.float32)  # fmt: skip

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        q = self.encode([self.q_prefix + x for x in queries])
        return [_top(row, k) for row in q @ self.emb.T]


# ---------------------------------------------------------------------------- hybrid


class Hybrid:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (c + rank)."""

    def __init__(self, retrievers: Sequence[Retriever], c: int = 60, depth: int = 100) -> None:
        self.retrievers, self.c, self.depth = list(retrievers), c, depth
        self.name = "+".join(r.name for r in self.retrievers)

    def search(self, queries: Sequence[str], k: int) -> list[Ranking]:
        runs = [r.search(queries, self.depth) for r in self.retrievers]
        out = []
        for per_query in zip(*runs, strict=True):
            fused: dict[int, float] = {}
            for ranking in per_query:
                for rank, (i, _) in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1 / (self.c + rank + 1)
            out.append(sorted(fused.items(), key=lambda x: -x[1])[:k])
        return out


def _top(scores: np.ndarray, k: int) -> Ranking:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def norms(ranking: Ranking, passages: Sequence[Passage]) -> list[str]:
    """Distinct norms in rank order: a norm ranks where its best passage ranks."""
    return list(dict.fromkeys(passages[i].norm_id for i, _ in ranking))
