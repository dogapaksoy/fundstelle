| retriever | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | s / question |
|---|---|---|---|---|---|---|---|
| bm25 | 5.0% | 12.0% | 15.5% | 18.5% | 13.5% | 0.083 | 0.00 s/q |
| dense:intfloat/multilingual-e5-small | 4.5% | 11.5% | 13.5% | 22.0% | 11.3% | 0.085 | 0.05 s/q |
| dense:models/e5-small-gerlayqa | 16.5% | 25.5% | 32.5% | 43.5% | 26.8% | 0.224 | 0.04 s/q |
| bm25+dense:models/e5-small-gerlayqa | 12.0% | 24.0% | 31.0% | 37.5% | 25.8% | 0.187 | 0.05 s/q |
