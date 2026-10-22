# Part 2 — SD-QA real speech results

747 answerable TyDi-QA English questions; corpus of 46,185 Wikipedia passages from 929 articles. Speech: real speakers, 11 English dialects; transcripts from Google ASR with a localised model and the en-US model (released by SD-QA). Re-ranking: BM25 top-25. Fine-tuned models: protocol B (fixed 3 epochs), per-query scores averaged over seeds 0–2. Scoring in fp16.


## 1. Degradation by dialect and ASR model (nDCG@10)

| Dialect | ASR model | WER | BM25 | Dense | Hybrid | BM25→CE | BM25→CE vs typed |
|---|---|---|---|---|---|---|---|
| *Typed (gold)* | — | 0.000 | 0.3749 | 0.6150 | 0.5293 | 0.5813 | — |
| US | en-US | 0.042 | 0.3603 | 0.5945 | 0.5198 | 0.5622 | -3.3% |
| Kenya | en-KE | 0.154 | 0.3108 | 0.5170 | 0.4527 | 0.4884 | -16.0% |
| Kenya | en-US | 0.177 | 0.2904 | 0.4809 | 0.4241 | 0.4678 | -19.5% |
| UK | en-US | 0.109 | 0.3182 | 0.5243 | 0.4653 | 0.5078 | -12.6% |
| UK | en-GB | 0.080 | 0.3360 | 0.5581 | 0.4925 | 0.5432 | -6.6% |
| Australia | en-US | 0.102 | 0.3220 | 0.5342 | 0.4636 | 0.5126 | -11.8% |
| Australia | en-AU | 0.082 | 0.3405 | 0.5632 | 0.4976 | 0.5492 | -5.5% |
| Ireland | en-US | 0.086 | 0.3349 | 0.5533 | 0.4897 | 0.5324 | -8.4% |
| Ireland | en-IE | 0.083 | 0.3291 | 0.5509 | 0.4930 | 0.5353 | -7.9% |
| Philippines | en-PH | 0.084 | 0.3323 | 0.5451 | 0.4840 | 0.5382 | -7.4% |
| Philippines | en-US | 0.076 | 0.3305 | 0.5618 | 0.4842 | 0.5351 | -7.9% |
| Nigeria | en-US | 0.156 | 0.2894 | 0.4922 | 0.4319 | 0.4712 | -18.9% |
| Nigeria | en-NG | 0.118 | 0.3086 | 0.5188 | 0.4586 | 0.5048 | -13.2% |
| South Africa | en-US | 0.078 | 0.3467 | 0.5610 | 0.4986 | 0.5463 | -6.0% |
| South Africa | en-ZA | 0.069 | 0.3536 | 0.5760 | 0.5101 | 0.5563 | -4.3% |
| India-North | en-US | 0.102 | 0.3218 | 0.5441 | 0.4719 | 0.5101 | -12.3% |
| India-North | en-IN | 0.129 | 0.3034 | 0.4955 | 0.4436 | 0.4901 | -15.7% |
| India-South | en-IN | 0.176 | 0.2845 | 0.4614 | 0.4138 | 0.4537 | -21.9% |
| India-South | en-US | 0.120 | 0.3046 | 0.5174 | 0.4529 | 0.4794 | -17.5% |
| New Zealand | en-NZ | 0.067 | 0.3377 | 0.5715 | 0.4934 | 0.5363 | -7.7% |
| New Zealand | en-US | 0.089 | 0.3277 | 0.5447 | 0.4792 | 0.5180 | -10.9% |

Mean BM25→CE change vs typed: all conditions -11.2%, Indian-English conditions -16.9%.


## 2. Fine-tuned rerankers on real speech (nDCG@10; per-query scores averaged over the conditions in each group)

Significance: paired randomization test on the group-averaged per-query nDCG@10, Holm-corrected within each group. † p<0.05 vs zero-shot; ‡ p<0.05 vs FT-clean.

Last column: uncorrected p for FT-distill vs zero-shot.

| Group | Zero-shot | FT-clean | FT-noisy | FT-mixed | FT-distill | distill vs zero-shot, raw p |
|---|---|---|---|---|---|---|
| Typed (gold) | 0.5813 | 0.5817 | 0.5796 | 0.5814 | 0.5859 | 0.0302 |
| All 21 real-speech conditions | 0.5161 | 0.5151 | 0.5163 | 0.5163 | 0.5185 | 0.0793 |
| Indian English (4 conditions) | 0.4833 | 0.4817 | 0.4832 | 0.4835 | 0.4863 | 0.0289 |
| Indian English, Indian ASR model | 0.4719 | 0.4697 | 0.4709 | 0.4713 | 0.4754 | 0.0156 |
| US English | 0.5622 | 0.5636 | 0.5636 | 0.5642 | 0.5641 | 0.2787 |

## 3. Localised vs en-US recogniser, same speakers (BM25→CE zero-shot nDCG@10)

| Dialect | Localised model | Localised | en-US | Δ (localised − US) |
|---|---|---|---|---|
| Kenya | en-KE | 0.4884 | 0.4678 | +0.0207 * |
| UK | en-GB | 0.5432 | 0.5078 | +0.0354 * |
| Australia | en-AU | 0.5492 | 0.5126 | +0.0366 * |
| Ireland | en-IE | 0.5353 | 0.5324 | +0.0029 |
| Philippines | en-PH | 0.5382 | 0.5351 | +0.0030 |
| Nigeria | en-NG | 0.5048 | 0.4712 | +0.0336 * |
| South Africa | en-ZA | 0.5563 | 0.5463 | +0.0100 |
| India-North | en-IN | 0.4901 | 0.5101 | -0.0200 * |
| India-South | en-IN | 0.4537 | 0.4794 | -0.0257 * |
| New Zealand | en-NZ | 0.5363 | 0.5180 | +0.0184 * |

\* p < 0.05 (paired randomization test, uncorrected).


## 4. Which first stage to re-rank (nDCG@10; Recall@100 of the first stage in brackets)

p-values: paired randomization test, uncorrected (exploratory; not pre-registered).

| Group | BM25 | Dense | Hybrid | BM25→CE | Hybrid→CE | Hybrid→FT-distill | Hybrid→CE vs Dense p | Hybrid→distill vs Hybrid→CE p |
|---|---|---|---|---|---|---|---|---|
| Typed (gold) | 0.3749 (0.897) | 0.6150 (0.987) | 0.5293 (0.985) | 0.5813 | 0.6712 | 0.6770 | 0.0001 | 0.0106 |
| All 21 real-speech conditions | 0.3230 (0.805) | 0.5365 (0.902) | 0.4724 (0.902) | 0.5161 | 0.6073 | 0.6111 | 0.0001 | 0.0270 |
| Indian English (4 conditions) | 0.3036 (0.771) | 0.5046 (0.866) | 0.4456 (0.869) | 0.4833 | 0.5718 | 0.5760 | 0.0001 | 0.0181 |
| Indian English, Indian ASR model | 0.2939 (0.752) | 0.4784 (0.839) | 0.4287 (0.846) | 0.4719 | 0.5521 | 0.5569 | 0.0001 | 0.0107 |
| US English | 0.3603 (0.855) | 0.5945 (0.967) | 0.5198 (0.963) | 0.5622 | 0.6594 | 0.6641 | 0.0001 | 0.0282 |
