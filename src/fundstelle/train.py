"""Fine-tune an embedding model on lay questions and the norms lawyers cited for them.

Off-the-shelf embeddings place "Mein Vermieter will mich rauswerfen, weil er die Wohnung
für seine Tochter braucht" far from "§ 573 Ordentliche Kündigung des Vermieters", because
nothing in their pre-training connects a story to the statute that governs it. GerLayQA
has 23k such questions outside the evaluation splits, each labelled with the norms a
lawyer cited. That is a contrastive training set: pull each question towards its cited
norms, away from the other norms in the batch and from a hard negative, a norm BM25
ranks high that the lawyer did not cite.
"""

from __future__ import annotations

import random
from pathlib import Path

from fundstelle.corpus import Passage, norm_docs
from fundstelle.data import FILES, Question, load_raw
from fundstelle.retrieval import BM25, PREFIXES, norms


def training_questions(data: Path, held_out: list[Question], norm_ids: set[str]) -> list[Question]:
    held = {q.text for q in held_out}
    out, seen = [], set()
    for law in FILES:
        for q in load_raw(data, law):
            if q.text in held or q.text in seen or not q.gold:
                continue
            if all(g in norm_ids for g in q.gold):
                seen.add(q.text)
                out.append(q)
    return out


def triplets(
    questions: list[Question],
    passages: list[Passage],
    seed: int = 0,
    negatives_from: int = 30,
    per_question: int | None = None,
) -> list[tuple[str, str, str]]:
    """(question, cited norm, hard negative norm) for the cited norms of every question
    (at most ``per_question`` of them, to fit a CPU budget)."""
    docs = norm_docs(passages)
    by_id = {d.id: d for d in docs}
    bm = BM25(docs)
    rng = random.Random(seed)
    out = []
    rankings = bm.search([q.text for q in questions], negatives_from)
    for q, ranking in zip(questions, rankings, strict=True):
        candidates = [n for n in norms(ranking, docs) if n not in q.gold]
        for g in q.gold[:per_question]:
            neg = rng.choice(candidates) if candidates else rng.choice(docs).id
            out.append((q.text, by_id[g].for_index(), by_id[neg].for_index()))
    rng.shuffle(out)
    return out


def one_per_question(triples: list[tuple[str, str, str]], n: int, seed: int = 0):
    """One triplet per question (a random one of its cited norms), ``n`` questions."""
    seen, out = set(), []
    for t in triples:
        if t[0] not in seen:
            seen.add(t[0])
            out.append(t)
    random.Random(seed).shuffle(out)
    return out[:n]


def train(triples: list[tuple[str, str, str]], base: str, out: Path, epochs: int = 1,
          batch_size: int = 32, max_length: int = 256, lr: float = 3e-5) -> Path:  # fmt: skip
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    q_prefix, p_prefix = PREFIXES.get(base, ("", ""))
    model = SentenceTransformer(base, device="cpu")
    model.max_seq_length = max_length
    # The multilingual vocabulary (250k tokens) is most of the parameters; training it
    # would need gigabytes of optimizer state for words a legal corpus barely uses.
    for name, param in model.named_parameters():
        if "word_embeddings" in name:
            param.requires_grad = False
    ds = Dataset.from_dict({
        "anchor": [q_prefix + a for a, _, _ in triples],
        "positive": [p_prefix + p for _, p, _ in triples],
        "negative": [p_prefix + n for _, _, n in triples],
    })  # fmt: skip
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # a norm twice in a batch is a false negative
        logging_steps=20, save_strategy="no", report_to=[], use_cpu=True, seed=0,
    )  # fmt: skip
    loss = losses.MultipleNegativesRankingLoss(model)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(str(out))
    return out
