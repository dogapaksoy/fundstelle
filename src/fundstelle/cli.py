"""Command line: download data, evaluate retrievers, fine-tune an embedding model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fundstelle.corpus import download, load
from fundstelle.data import splits


def _corpus(data: Path):
    path = data / "corpus.jsonl"
    if not path.exists():
        download(data)
    return load(path)


def _dense(model: str, docs, data: Path):
    from fundstelle.retrieval import Dense

    prefixes = None
    meta = Path(model) / "fundstelle.json"
    if meta.exists():  # a model fine-tuned by `fundstelle train` records its prefixes
        prefixes = tuple(json.loads(meta.read_text())["prefixes"])
    return Dense(docs, model, cache=data / "cache", prefixes=prefixes)


class _ByNorm:
    """A retriever over whole-norm documents, reporting each norm's first Absatz."""

    def __init__(self, inner, docs, passages) -> None:
        first: dict[str, int] = {}
        for i, p in enumerate(passages):
            first.setdefault(p.norm_id, i)
        self.inner, self.name = inner, inner.name.replace("dense:", "norms:")
        self.ids = [first[d.id] for d in docs]

    def search(self, queries, k):
        return [[(self.ids[i], s) for i, s in r] for r in self.inner.search(queries, k)]


def build(name: str, passages, data: Path):
    """Retrievers joined by ``+`` are fused per norm (reciprocal rank fusion); a part can
    carry a weight, ``2*cases:<model>``. Parts:

    - ``bm25``
    - ``dense:<model>``: embeddings of Absätze
    - ``norms:<model>``: embeddings of whole norms (how ``fundstelle train`` trains)
    - ``cases:<model>``: the 30 most similar training questions vote for their cited norms
    """
    from fundstelle.corpus import norm_docs
    from fundstelle.retrieval import BM25, Cases, Hybrid

    parts, weights = [], []
    for part in name.split("+"):
        weight, _, part = part.rpartition("*")
        weights.append(float(weight) if weight else 1.0)
        kind, _, model = part.partition(":")
        if kind == "bm25":
            parts.append(BM25(passages))
        elif kind == "dense":
            parts.append(_dense(model, passages, data))
        elif kind == "norms":
            docs = norm_docs(passages)
            parts.append(_ByNorm(_dense(model, docs, data), docs, passages))
        elif kind == "cases":
            from fundstelle.train import training_questions

            norm_ids = {p.norm_id for p in passages}
            held = [q for qs in splits(data, norm_ids).values() for q in qs]
            questions = training_questions(data, held, norm_ids)
            parts.append(Cases(questions, _dense(model, passages, data), passages, data / "cache"))
        else:
            raise ValueError(f"unknown retriever {part!r}")
    if len(parts) == 1:
        return parts[0]
    return Hybrid(parts, weights=weights, passages=passages)


def cmd_eval(args: argparse.Namespace) -> None:
    from fundstelle.evaluate import header, run, score

    passages = _corpus(args.data)
    questions = splits(args.data, {p.norm_id for p in passages})[args.split]
    lines = [header(timing=True)]
    for name in args.retrievers:
        r = build(name, passages, args.data)
        t0 = time.perf_counter()
        rankings = r.search([q.text for q in questions], 100)
        per_q = (time.perf_counter() - t0) / len(questions)
        safe = name.replace("/", "_").replace(":", "-")
        rows = run(name, rankings, questions, passages, args.out / f"{args.split}-{safe}.jsonl")
        lines.append(score(name, rows).row() + f" {per_q:.2f} s/q |")
        print(lines[-1], flush=True)
    path = args.out / f"{args.split}-summary.md"
    # keep rows of retrievers evaluated earlier; replace those evaluated again
    names = {line.split(" | ")[0] for line in lines[1:]}
    if path.exists():
        old = [
            x
            for x in path.read_text(encoding="utf-8").splitlines()[2:]
            if x.split(" | ")[0] not in names
        ]
        lines = lines[:1] + old + lines[1:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> None:
    from fundstelle.retrieval import PREFIXES
    from fundstelle.train import one_per_question, train, training_questions, triplets

    passages = _corpus(args.data)
    norm_ids = {p.norm_id for p in passages}
    held = [q for qs in splits(args.data, norm_ids).values() for q in qs]
    questions = training_questions(args.data, held, norm_ids)
    triples = one_per_question(triplets(questions, passages), args.questions)
    train(triples, args.base, args.out, batch_size=args.batch_size, max_length=args.max_length)
    meta = {"base": args.base, "prefixes": PREFIXES.get(args.base, ("", "")),
            "questions": len(triples)}  # fmt: skip
    (args.out / "fundstelle.json").write_text(json.dumps(meta), encoding="utf-8")


def cmd_answers(args: argparse.Namespace) -> None:
    from fundstelle.answer import Answerer, NLISupport
    from fundstelle.evaluate_answers import HEADER, run, summary
    from fundstelle.llm import LlamaCppBackend

    passages = _corpus(args.data)
    test = splits(args.data, {p.norm_id for p in passages})[args.split]
    per_law = {"BGB": args.n // 2, "StGB": args.n // 4, "ZPO": args.n - args.n // 2 - args.n // 4}
    questions = [q for law, k in per_law.items() for q in [x for x in test if x.law == law][:k]]
    retriever = build(args.retriever, passages, args.data)
    rankings = retriever.search([q.text for q in questions], 50)
    answerer = Answerer(LlamaCppBackend(args.model), NLISupport())
    lines = [HEADER]
    conditions = {"retrieved": (False, 0), "oracle": (True, 0), "norms": (False, args.norms)}
    for tag in args.conditions.split(","):
        oracle, norms = conditions[tag]
        shown = f"top {norms} norms, all Absätze" if norms else f"top {args.k} Absätze"
        name = f"{args.model}, {'lawyer norms shown first' if oracle else shown}"
        out = args.out / f"{args.split}-{args.model}-{tag}.jsonl"
        rows = run(name, answerer, questions, rankings, passages, out, k=args.k, oracle=oracle,
                   norms=norms)  # fmt: skip
        lines.append(summary(name, rows))
    (args.out / f"{args.split}-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_perturb(args: argparse.Namespace) -> None:
    from fundstelle.answer import NLISupport
    from fundstelle.perturb import run

    by_id = {p.id: p for p in _corpus(args.data)}
    rows = [json.loads(x) for f in args.answers for x in f.read_text(encoding="utf-8").splitlines()]
    result = run(rows, by_id, NLISupport())
    lines = ["| rule | tested | rejected by the check |", "|---|---|---|"]
    for kind, (n, rejected) in result["counts"].items():
        lines.append(f"| {kind} | {n} | {rejected / n:.0%} |" if n else f"| {kind} | 0 | – |")
    examples = json.dumps(result["examples"], ensure_ascii=False, indent=1)
    args.out.write_text("\n".join(lines) + "\n\n" + examples + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_download(args: argparse.Namespace) -> None:
    print(download(args.data))


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ.setdefault("FUNDSTELLE_DATA", str(args.data))
    uvicorn.run("fundstelle.service:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fundstelle")
    ap.add_argument("--data", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="retrieval metrics on GerLayQA (norms cited by lawyers)")
    e.add_argument("retrievers", nargs="+", help="e.g. bm25 dense:BAAI/bge-m3 bm25+dense:...")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--out", type=Path, default=Path("results/retrieval"))
    e.set_defaults(func=cmd_eval)

    t = sub.add_parser("train", help="fine-tune an embedding model on GerLayQA training data")
    t.add_argument("--base", default="intfloat/multilingual-e5-small")
    t.add_argument("--out", type=Path, default=Path("models/e5-small-gerlayqa"))
    t.add_argument("--questions", type=int, default=10000)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--max-length", type=int, default=192)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("answers", help="end-to-end cited answers, with and without oracle norms")
    a.add_argument("--retriever", default="dense:models/e5-small-gerlayqa")
    a.add_argument("--model", default="qwen3-4b")
    a.add_argument("--split", choices=["dev", "test"], default="test")
    a.add_argument("-n", type=int, default=60, help="questions: half BGB, a quarter StGB, ZPO")
    a.add_argument("-k", type=int, default=8, help="passages shown to the model")
    a.add_argument("--norms", type=int, default=4, help="norms shown in the `norms` condition")
    a.add_argument("--conditions", default="retrieved,oracle,norms")
    a.add_argument("--out", type=Path, default=Path("results/answers"))
    a.set_defaults(func=cmd_answers)

    pt = sub.add_parser("perturb", help="break accepted rules; how many does the check catch?")
    pt.add_argument("answers", nargs="+", type=Path, help="jsonl files from `answers`")
    pt.add_argument("--out", type=Path, default=Path("results/answers/perturbations.md"))
    pt.set_defaults(func=cmd_perturb)

    d = sub.add_parser("download", help="fetch BGB, StGB and ZPO from gesetze-im-internet.de")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("serve", help="run the HTTP service (see fundstelle.service)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
