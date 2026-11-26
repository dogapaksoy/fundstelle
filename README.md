# fundstelle

**Finds the statute a lawyer would cite for a layperson's legal question, and answers
only with what that statute says.**

People describe legal problems as stories: *"Mein Vermieter will mich rauswerfen, weil
seine Tochter einziehen soll."* Statutes are written in another language: *"ordentliche
Kündigung des Vermieters, berechtigtes Interesse"*. `fundstelle` (German for the exact
place a citation points to) bridges the two for the German Civil Code (BGB), Criminal
Code (StGB) and Code of Civil Procedure (ZPO), and runs on a laptop CPU.

It is evaluated on **real questions**: 1,000 held-out questions that people posted to an
online legal advice forum, each answered by a lawyer who cited the norms they relied on
([GerLayQA](https://huggingface.co/datasets/ViolaCamille/GerLayQA), Büttner & Habernal,
2024). The statute texts are the current official XML from
[gesetze-im-internet.de](https://www.gesetze-im-internet.de).

## Results

### Retrieval: is the lawyer's norm among the results?

1,000 test questions (600 BGB, 200 StGB, 200 ZPO), 8,688 passages (one per Absatz) from
4,055 norms. *hit@10*: a norm the lawyer cited is among the first 10 distinct norms.
4 CPU cores.

| retriever | hit@1 | hit@5 | **hit@10** | hit@20 | MRR | s / question |
|---|---|---|---|---|---|---|
| BM25 (German stemming) | 3.2% | 8.8% | 13.0% | 17.7% | 0.064 | 0.00 |
| multilingual-e5-small | 4.6% | 12.1% | 17.3% | 23.2% | 0.086 | 0.05 |
| bge-m3 (568M) | 6.1% | 16.0% | 23.3% | 29.8% | 0.114 | 0.43 |
| BM25 + bge-m3 (rank fusion) | 5.3% | 15.1% | 21.8% | 29.0% | 0.106 | 0.44 |
| **e5-small, fine-tuned on GerLayQA** | **11.3%** | **28.0%** | **37.8%** | **45.8%** | **0.197** | 0.05 |
| BM25 + fine-tuned e5-small | 10.3% | 21.9% | 31.1% | 41.2% | 0.168 | 0.04 |

- **The task is hard for off-the-shelf retrieval.** Questions are ~1,100 characters of
  story; BM25 finds the lawyer's norm in the top 10 for 13% of them, a 568M multilingual
  embedding model for 23%.
- **Fine-tuning a 118M model on 10,000 question → cited-norm pairs more than doubles
  hit@10 (17.3% → 37.8%)** and beats bge-m3, a model five times its size, by 14.5 points
  (95% bootstrap CI 11.6–17.5) at a ninth of the latency. Training took 85 minutes on
  4 CPU cores.
- **Rank fusion with BM25 hurts the fine-tuned model** (37.8% → 31.1%): once the
  embeddings know the domain, BM25 mostly adds noise from shared story words.
- **ZPO stays weak** (18% vs. 43% for BGB and StGB). It has the fewest training
  questions (~1,300 of 23,000); civil procedure questions also tend to cite general
  procedural norms that match many stories.

hit@10 is a lower bound: a lawyer cites one or two norms, and other norms in the top 10
are often also relevant. All per-question rankings are in
[`results/retrieval/`](results/retrieval).

### Answers: what reaches the user

40 test questions (20 BGB, 10 StGB, 10 ZPO), Qwen3-4B (Q4, local), top 8 passages of
the fine-tuned retriever. *Oracle* puts the passages of the lawyer's norms first, to
separate retrieval errors from generation errors.

| condition | lawyer's norm shown | answer cites lawyer's norm | rules the check accepts | no answer | s / answer |
|---|---|---|---|---|---|
| retrieved passages | 32% | 25% | 97% | 10% | 145 |
| oracle: lawyer's norms shown first | 100% | 62% | 92% | 15% | 166 |

- **Retrieval is the bottleneck.** When the lawyer's norm is among the passages, the model
  cites it in most cases (10 of 13 questions with retrieval, 62% with the oracle). Without
  it, it can't.
- **The model says when nothing fits.** Every "no answer" (10–15%) is the model returning
  no points because none of the passages matched the question, not the check rejecting
  everything.

**Does the check catch wrong rules?** Accepted rules were broken on purpose and checked
again ([`fundstelle perturb`](src/fundstelle/perturb.py)):

| rule | tested | rejected |
|---|---|---|
| negated ("kann" → "kann nicht", "nur" → "nicht") | 183 | 92% |
| a number changed ("zwei Wochen" → "vier Wochen") | 33 | 100% |
| correct rule, cited to another passage shown | 255 | 91% |

**What the check misses.** A hand audit of 20 accepted rules
([`results/answers/audit.md`](results/answers/audit.md)) found 15 correct, 4 simplified (a
condition dropped: "Die Veränderung von amtlichen Ausweisen ist strafbar" without "zur
Täuschung im Rechtsverkehr"), and 1 wrong: for § 637 BGB the model wrote that the customer
may fix a defect *if the contractor refuses*, where the statute says *unless the
contractor rightly refuses*, after a deadline. NLI scored it 0.99. Sentence-level NLI
catches flipped and swapped statements, not subtly inverted conditions or dropped ones;
that is why the passage text is always shown next to the rule.

## How it works

```
question ──► fine-tuned e5-small ──► top 8 Absätze ("BGB § 573 Abs. 2", title, text)
                                         │
                                         ▼
                   LLM (Qwen3-4B, local), JSON schema: up to 4 points
                   {cite: one of the 8 ids, rule: what it says, relevance: why it matters}
                                         │
                                         ▼  check, before anything is shown
                   NLI (multilingual, via belegt): does the cited Absatz entail `rule`?
                                         │
                     no ──► point dropped │ yes ──► "§ 573 Abs. 2 BGB: <rule>
                                         │          Einschätzung: <relevance>"
                     nothing left ──► "Die gefundenen Gesetzesstellen beantworten
                                       Ihre Frage nicht sicher." + where to get advice
```

Design decisions, each from a failure:

- **Passages are Absätze, labelled the way lawyers cite them.** "§ 573 Abs. 2 BGB" is
  both the retrieval unit and the citation the user sees. The XML parser keeps list items
  ("1. der Mieter ...") and drops repealed norms.
- **Questions that name their answer are excluded from evaluation.** 3–5% of questions
  already mention the cited paragraph ("gilt § 573 hier?"); for them retrieval is string
  matching.
- **Training data never overlaps the evaluation.** Dev and test questions are sampled
  first; fine-tuning uses the remaining 23k. Hard negatives are norms BM25 ranks high that
  the lawyer did not cite; the multilingual word embeddings are frozen (they are most of
  the parameters and a legal corpus uses few of the 250k tokens), which keeps training in
  9 GB of RAM.
- **Citations can't be invented.** The answer schema only allows the ids of the passages
  shown, so a citation can be misapplied but never made up.
- **Check the rule, label the reading.** A first version checked whole answer sentences
  against the cited text, and almost nothing passed (median entailment 0.01): the model
  mixes the rule with its application to the user's story ("Die Vereinbarung kann
  anfechtbar sein, wenn die Mieter getäuscht haben"), which no statute entails. Now each
  point restates one Absatz (checked) and says separately why it matters (shown as
  *Einschätzung*, the model's assessment).

## Service

`fundstelle serve` runs a FastAPI service with a small web page:

- `POST /search {"question": ...}`: the top passages with citations and scores
- `POST /ask {"question": ...}`: checked points, the dropped ones, the passages, and a
  trace (retrieval / model / check seconds, points drafted and kept)
- `GET /metrics`: request counts and seconds in Prometheus text format; every request
  is also logged as one JSON line

The model is configurable: a local GGUF model (`FUNDSTELLE_LLM=llama:qwen3-4b`) or any
OpenAI-compatible server (`FUNDSTELLE_LLM=openai:<model>@<base url>`).

```bash
docker build -t fundstelle .
docker run -p 8000:8000 fundstelle    # then open http://localhost:8000
```

## Usage

```bash
pip install "fundstelle[retrieval,answer,serve] @ git+https://github.com/dogapaksoy/fundstelle.git"
fundstelle download                       # statutes from gesetze-im-internet.de
fundstelle eval --split test bm25 dense:BAAI/bge-m3
fundstelle train                          # fine-tune e5-small (~1.5 h on 4 cores)
fundstelle eval --split test dense:models/e5-small-gerlayqa
fundstelle answers -n 40                  # end-to-end answers with Qwen3-4B
fundstelle serve
```

## Limitations

- **The check is not a proof.** It catches nine in ten negated or misattributed rules,
  but a dropped or inverted condition can pass (see the audit). The statute text is
  always shown next to each rule.
- **Not legal advice.** The assistant restates statutes and points to them. Whether a
  rule applies to a case depends on facts and case law it does not know; the
  *Einschätzung* is a model's guess and is labelled as such.
- **Three laws.** BGB, StGB, ZPO, because GerLayQA covers them. Questions about
  residence law or social benefits need other statutes and other training data.
- **Gold labels are partial.** A lawyer cites the norms they needed for their answer, not
  every relevant norm. Retrieval and citation scores against them are lower bounds.
- **Statutes change, questions are older.** The questions are from past years; the
  statute text is current. A few gold norms (1%) no longer exist and those questions are
  excluded.
- **CPU latency.** Retrieval takes 0.05 s; a local 4B model takes 2–3 minutes per answer
  on 4 cores. A deployment wants a GPU or a hosted model (the service supports both).

## License

MIT. GerLayQA is Apache-2.0. Statute texts are official works and not subject to
copyright (§ 5 UrhG).
