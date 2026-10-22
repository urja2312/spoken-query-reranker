# Part 3 error analysis (pre-registered C4 and C5)

Δ = nDCG@10(spoken) − nDCG@10(typed) for BM25 > ce-l6 @25. Content words: NOUN, PROPN, NUM, ADJ, non-auxiliary VERB. Bootstrap: 2,000 resamples of queries (clusters).


## C4. Which words break

| Group | Pairs | Content error rate | Function error rate | β content [95% CI] | β function [95% CI] | β content − β function [95% CI] |
|---|---|---|---|---|---|---|
| SD-QA, Google ASR | 14784 | 0.121 | 0.058 | -0.646 [-0.745, -0.553] | +0.117 [+0.034, +0.213] | -0.763 [-0.923, -0.611] |
| BEIR, wideband | 1046 | 0.130 | 0.060 | -0.369 [-0.498, -0.252] | -0.025 [-0.134, +0.069] | -0.344 [-0.522, -0.168] |
| BEIR, telephone | 1046 | 0.315 | 0.129 | -0.363 [-0.443, -0.285] | -0.011 [-0.086, +0.061] | -0.352 [-0.475, -0.241] |

Named entities (secondary): mean Δ when any entity word is misrecognised vs when none is.

| Group | Δ, entity word wrong | Δ, entity words right |
|---|---|---|
| SD-QA, Google ASR | -0.267 | -0.007 |
| BEIR, wideband | -0.140 | -0.016 |
| BEIR, telephone | -0.207 | -0.041 |

## C5. Where the loss happens

Share of (query, condition) pairs that lose nDCG@10 in which no relevant passage is left in the BM25 top-25 of the spoken query (the reranker never sees it).

| Group | Pairs that lose | Recall-failure share [95% CI] |
|---|---|---|
| SD-QA, Google ASR | 2668 | 0.532 [0.468, 0.594] |
| BEIR, wideband | 185 | 0.422 [0.351, 0.497] |
| BEIR, telephone | 390 | 0.556 [0.505, 0.605] |
