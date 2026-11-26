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


def build(name: str, passages, data: Path):
    """``bm25``, ``dense:<model>``, or several joined by ``+`` (reciprocal rank fusion)."""
    from fundstelle.retrieval import BM25, Dense, Hybrid

    parts = []
    for part in name.split("+"):
        if part == "bm25":
            parts.append(BM25(passages))
        elif part.startswith("dense:"):
            model = part.removeprefix("dense:")
            prefixes = None
            meta = Path(model) / "fundstelle.json"
            if meta.exists():  # a model fine-tuned by `fundstelle train` records its prefixes
                prefixes = tuple(json.loads(meta.read_text())["prefixes"])
            parts.append(Dense(passages, model, cache=data / "cache", prefixes=prefixes))
        else:
            raise ValueError(f"unknown retriever {part!r}")
    return parts[0] if len(parts) == 1 else Hybrid(parts)


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
    (args.out / f"{args.split}-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
