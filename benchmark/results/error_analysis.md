# Part 3 error analysis (pre-registered C4 and C5)

Δ = nDCG@10(spoken) − nDCG@10(typed) for BM25 > ce-l6 @25. Content words: NOUN, PROPN, NUM, ADJ, non-auxiliary VERB. Bootstrap: 2,000 resamples of queries (clusters).


## C4. Which words break

| Group | Pairs | Content error rate | Function error rate | β content [95% CI] | β function [95% CI] | β content − β function [95% CI] |
|---|---|---|---|---|---|---|
| SD-QA, Whisper base.en | 7744 | 0.181 | 0.085 | -0.639 [-0.716, -0.569] | +0.046 [-0.016, +0.111] | -0.685 [-0.812, -0.575] |
| SD-QA, Google ASR | 14784 | 0.121 | 0.058 | -0.646 [-0.745, -0.553] | +0.117 [+0.034, +0.213] | -0.763 [-0.923, -0.611] |
| BEIR, wideband | 1046 | 0.130 | 0.060 | -0.368 [-0.495, -0.251] | -0.025 [-0.134, +0.068] | -0.343 [-0.520, -0.166] |
| BEIR, telephone | 1046 | 0.315 | 0.129 | -0.364 [-0.443, -0.286] | -0.010 [-0.085, +0.063] | -0.353 [-0.477, -0.242] |

Named entities (secondary): mean Δ when any entity word is misrecognised vs when none is.

| Group | Δ, entity word wrong | Δ, entity words right |
|---|---|---|
| SD-QA, Whisper base.en | -0.284 | -0.019 |
| SD-QA, Google ASR | -0.267 | -0.007 |
| BEIR, wideband | -0.140 | -0.016 |
| BEIR, telephone | -0.207 | -0.042 |

## C5. Where the loss happens

Share of (query, condition) pairs that lose nDCG@10 in which no relevant passage is left in the BM25 top-25 of the spoken query (the reranker never sees it).

| Group | Pairs that lose | Recall-failure share [95% CI] |
|---|---|---|
| SD-QA, Whisper base.en | 1655 | 0.682 [0.637, 0.722] |
| SD-QA, Google ASR | 2668 | 0.532 [0.468, 0.594] |
| BEIR, wideband | 185 | 0.422 [0.351, 0.497] |
| BEIR, telephone | 391 | 0.555 [0.504, 0.606] |
