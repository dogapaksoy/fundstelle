| retriever | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | s / question |
|---|---|---|---|---|---|---|---|
| bm25 | 3.2% | 8.8% | 13.0% | 17.7% | 10.2% | 0.064 | 0.00 s/q |
| dense:intfloat/multilingual-e5-small | 4.6% | 12.1% | 17.3% | 23.2% | 13.5% | 0.086 | 0.05 s/q |
| dense:models/e5-small-gerlayqa | 11.3% | 28.0% | 37.8% | 45.8% | 31.0% | 0.197 | 0.05 s/q |
| bm25+dense:models/e5-small-gerlayqa | 10.3% | 21.9% | 31.1% | 41.2% | 24.9% | 0.168 | 0.04 s/q |
| dense:BAAI/bge-m3 | 6.1% | 16.0% | 23.3% | 29.8% | 18.5% | 0.114 | 0.43 s/q |
| bm25+dense:BAAI/bge-m3 | 5.3% | 15.1% | 21.8% | 29.0% | 16.8% | 0.106 | 0.44 s/q |
| dense:models/e5-small-gerlayqa-full | 14.7% | 31.0% | 41.9% | 51.3% | 35.0% | 0.230 | 0.04 s/q |
| norms:models/e5-small-gerlayqa-full | 15.5% | 34.2% | 43.6% | 53.0% | 37.0% | 0.248 | 0.04 s/q |
| cases:models/e5-small-gerlayqa-full | 23.9% | 48.6% | 57.7% | 66.0% | 51.0% | 0.352 | 0.04 s/q |
| dense:models/e5-small-gerlayqa-full+2*cases:models/e5-small-gerlayqa-full | 25.6% | 50.3% | 60.6% | 67.3% | 53.9% | 0.373 | 0.08 s/q |
| dense:models/e5-small-gerlayqa-full+norms:models/e5-small-gerlayqa-full+2*cases:models/e5-small-gerlayqa-full | 25.4% | 49.8% | 60.6% | 70.3% | 53.7% | 0.370 | 0.12 s/q |
