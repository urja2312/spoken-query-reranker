# Reranker benchmark — results

Significance: two-sided paired randomization test on per-query nDCG@10 vs BM25, Holm-corrected within each dataset × condition; * p<0.05, ** p<0.01. CI: 95% bootstrap over queries.


### scifact — Text (oracle) (300 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.6656 | [0.619, 0.711] | 0.6315 | 0.5433 | 0.8759 | +0.0000 | — |
| Dense (bge-small) | 0.7027 | [0.657, 0.746] | 0.6729 | 0.6000 | 0.9467 | +0.0371 | 0.306 |
| Hybrid RRF | 0.7057** | [0.661, 0.748] | 0.6717 | 0.5867 | 0.9683 | +0.0401 | 0.001 |
| BM25 > ce-l6 @25 | 0.6737 | [0.627, 0.719] | 0.6484 | 0.5633 | 0.8759 | +0.0080 | 1.000 |
| BM25 > ce-l6 @100 | 0.6653 | [0.618, 0.711] | 0.6428 | 0.5600 | 0.8759 | -0.0003 | 1.000 |
| Hybrid > ce-l6 @25 | 0.6850 | [0.640, 0.729] | 0.6534 | 0.5667 | 0.9683 | +0.0194 | 1.000 |
| Hybrid > ce-l6 @100 | 0.6797 | [0.633, 0.724] | 0.6515 | 0.5667 | 0.9683 | +0.0141 | 1.000 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.6919 | [0.646, 0.736] | 0.6620 | 0.5800 | 0.8759 | +0.0262 | 0.064 |

### scifact — ASR wideband (300 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.5679 | [0.519, 0.616] | 0.5328 | 0.4467 | 0.7806 | +0.0000 | — |
| Dense (bge-small) | 0.6171* | [0.569, 0.663] | 0.5887 | 0.5067 | 0.8850 | +0.0493 | 0.029 |
| Hybrid RRF | 0.6100** | [0.562, 0.657] | 0.5791 | 0.5000 | 0.8967 | +0.0422 | 0.002 |
| BM25 > ce-l6 @25 | 0.5941 | [0.544, 0.643] | 0.5705 | 0.4900 | 0.7806 | +0.0262 | 0.051 |
| Hybrid > ce-l6 @25 | 0.6267** | [0.579, 0.673] | 0.5962 | 0.5067 | 0.8967 | +0.0589 | 0.002 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.6017** | [0.552, 0.650] | 0.5792 | 0.4967 | 0.7806 | +0.0338 | 0.002 |

### scifact — ASR telephone (300 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.4567 | [0.407, 0.504] | 0.4272 | 0.3500 | 0.7239 | +0.0000 | — |
| Dense (bge-small) | 0.5136** | [0.464, 0.563] | 0.4900 | 0.4167 | 0.7917 | +0.0569 | 0.010 |
| Hybrid RRF | 0.5325** | [0.484, 0.581] | 0.5023 | 0.4200 | 0.8100 | +0.0758 | <0.001 |
| BM25 > ce-l6 @25 | 0.5088** | [0.459, 0.559] | 0.4890 | 0.4200 | 0.7239 | +0.0521 | 0.001 |
| Hybrid > ce-l6 @25 | 0.5396** | [0.491, 0.588] | 0.5102 | 0.4233 | 0.8100 | +0.0829 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.5171** | [0.467, 0.567] | 0.4973 | 0.4300 | 0.7239 | +0.0604 | <0.001 |

### nfcorpus — Text (oracle) (323 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.3084 | [0.274, 0.342] | 0.5163 | 0.4365 | 0.2332 | +0.0000 | — |
| Dense (bge-small) | 0.3433** | [0.309, 0.378] | 0.5364 | 0.4520 | 0.3009 | +0.0349 | 0.003 |
| Hybrid RRF | 0.3523** | [0.318, 0.387] | 0.5690 | 0.4861 | 0.3107 | +0.0439 | <0.001 |
| BM25 > ce-l6 @25 | 0.3307** | [0.297, 0.364] | 0.5693 | 0.4985 | 0.2332 | +0.0223 | <0.001 |
| BM25 > ce-l6 @100 | 0.3383** | [0.304, 0.373] | 0.5693 | 0.5015 | 0.2332 | +0.0299 | <0.001 |
| Hybrid > ce-l6 @25 | 0.3549** | [0.320, 0.390] | 0.5794 | 0.5077 | 0.3107 | +0.0464 | <0.001 |
| Hybrid > ce-l6 @100 | 0.3504** | [0.315, 0.385] | 0.5786 | 0.5108 | 0.3107 | +0.0420 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.3283** | [0.295, 0.362] | 0.5600 | 0.4830 | 0.2332 | +0.0199 | <0.001 |

### nfcorpus — ASR wideband (323 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.2324 | [0.202, 0.263] | 0.3917 | 0.3096 | 0.1916 | +0.0000 | — |
| Dense (bge-small) | 0.2707** | [0.239, 0.303] | 0.4244 | 0.3313 | 0.2523 | +0.0383 | <0.001 |
| Hybrid RRF | 0.2778** | [0.246, 0.310] | 0.4536 | 0.3653 | 0.2610 | +0.0454 | <0.001 |
| BM25 > ce-l6 @25 | 0.2589** | [0.228, 0.291] | 0.4501 | 0.3653 | 0.1916 | +0.0265 | <0.001 |
| Hybrid > ce-l6 @25 | 0.2819** | [0.250, 0.314] | 0.4605 | 0.3684 | 0.2610 | +0.0495 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.2563** | [0.226, 0.288] | 0.4426 | 0.3560 | 0.1916 | +0.0239 | <0.001 |

### nfcorpus — ASR telephone (323 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.1291 | [0.106, 0.154] | 0.2332 | 0.1765 | 0.1364 | +0.0000 | — |
| Dense (bge-small) | 0.1653** | [0.138, 0.194] | 0.2795 | 0.2012 | 0.1835 | +0.0362 | <0.001 |
| Hybrid RRF | 0.1674** | [0.141, 0.195] | 0.2990 | 0.2384 | 0.1889 | +0.0383 | <0.001 |
| BM25 > ce-l6 @25 | 0.1574** | [0.131, 0.185] | 0.2894 | 0.2291 | 0.1364 | +0.0283 | <0.001 |
| Hybrid > ce-l6 @25 | 0.1844** | [0.156, 0.214] | 0.3160 | 0.2539 | 0.1889 | +0.0553 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.1555** | [0.129, 0.183] | 0.2844 | 0.2260 | 0.1364 | +0.0264 | <0.001 |

### fiqa — Text (oracle) (648 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.2328 | [0.209, 0.257] | 0.2910 | 0.2160 | 0.5176 | +0.0000 | — |
| Dense (bge-small) | 0.3913** | [0.363, 0.419] | 0.4764 | 0.3904 | 0.6830 | +0.1585 | <0.001 |
| Hybrid RRF | 0.3448** | [0.318, 0.371] | 0.4195 | 0.3287 | 0.6825 | +0.1120 | <0.001 |
| BM25 > ce-l6 @25 | 0.3046** | [0.278, 0.332] | 0.3871 | 0.3148 | 0.5176 | +0.0719 | <0.001 |
| BM25 > ce-l6 @100 | 0.3307** | [0.303, 0.358] | 0.4068 | 0.3225 | 0.5176 | +0.0979 | <0.001 |
| Hybrid > ce-l6 @25 | 0.3694** | [0.342, 0.397] | 0.4389 | 0.3410 | 0.6825 | +0.1366 | <0.001 |
| Hybrid > ce-l6 @100 | 0.3650** | [0.338, 0.392] | 0.4351 | 0.3364 | 0.6825 | +0.1322 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.3044** | [0.278, 0.331] | 0.3822 | 0.3071 | 0.5176 | +0.0716 | <0.001 |

### fiqa — ASR wideband (648 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.2141 | [0.191, 0.238] | 0.2715 | 0.2022 | 0.4842 | +0.0000 | — |
| Dense (bge-small) | 0.3636** | [0.336, 0.391] | 0.4446 | 0.3611 | 0.6536 | +0.1495 | <0.001 |
| Hybrid RRF | 0.3198** | [0.294, 0.347] | 0.3942 | 0.3102 | 0.6571 | +0.1057 | <0.001 |
| BM25 > ce-l6 @25 | 0.2824** | [0.256, 0.309] | 0.3664 | 0.3009 | 0.4842 | +0.0683 | <0.001 |
| Hybrid > ce-l6 @25 | 0.3455** | [0.318, 0.373] | 0.4160 | 0.3225 | 0.6571 | +0.1314 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.2837** | [0.257, 0.311] | 0.3659 | 0.3009 | 0.4842 | +0.0696 | <0.001 |

### fiqa — ASR telephone (648 queries)

| System | nDCG@10 | 95% CI | MRR@10 | Hit@1 | R@100 | Δ nDCG vs BM25 | p (Holm) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.1751 | [0.154, 0.197] | 0.2253 | 0.1698 | 0.4197 | +0.0000 | — |
| Dense (bge-small) | 0.2905** | [0.264, 0.318] | 0.3623 | 0.2886 | 0.5542 | +0.1154 | <0.001 |
| Hybrid RRF | 0.2617** | [0.237, 0.287] | 0.3226 | 0.2500 | 0.5650 | +0.0866 | <0.001 |
| BM25 > ce-l6 @25 | 0.2259** | [0.202, 0.251] | 0.3004 | 0.2469 | 0.4197 | +0.0508 | <0.001 |
| Hybrid > ce-l6 @25 | 0.2890** | [0.263, 0.316] | 0.3543 | 0.2762 | 0.5650 | +0.1139 | <0.001 |
| BM25 > ce-l6 @25 + alpha(CV) | 0.2257** | [0.202, 0.251] | 0.2984 | 0.2454 | 0.4197 | +0.0506 | <0.001 |

## Robustness to spoken queries (nDCG@10; relative change vs text)

| Dataset | System | Text | ASR wideband | ASR telephone | Δ wideband | Δ telephone |
|---|---|---|---|---|---|---|
| scifact | BM25 | 0.6656 | 0.5679 | 0.4567 | -14.7% | -31.4% |
| scifact | Dense (bge-small) | 0.7027 | 0.6171 | 0.5136 | -12.2% | -26.9% |
| scifact | Hybrid RRF | 0.7057 | 0.6100 | 0.5325 | -13.6% | -24.5% |
| scifact | BM25 > ce-l6 @25 | 0.6737 | 0.5941 | 0.5088 | -11.8% | -24.5% |
| scifact | BM25 > ce-l6 @100 | 0.6653 | — | — | — | — |
| nfcorpus | BM25 | 0.3084 | 0.2324 | 0.1291 | -24.7% | -58.1% |
| nfcorpus | Dense (bge-small) | 0.3433 | 0.2707 | 0.1653 | -21.2% | -51.9% |
| nfcorpus | Hybrid RRF | 0.3523 | 0.2778 | 0.1674 | -21.2% | -52.5% |
| nfcorpus | BM25 > ce-l6 @25 | 0.3307 | 0.2589 | 0.1574 | -21.7% | -52.4% |
| nfcorpus | BM25 > ce-l6 @100 | 0.3383 | — | — | — | — |
| fiqa | BM25 | 0.2328 | 0.2141 | 0.1751 | -8.0% | -24.8% |
| fiqa | Dense (bge-small) | 0.3913 | 0.3636 | 0.2905 | -7.1% | -25.8% |
| fiqa | Hybrid RRF | 0.3448 | 0.3198 | 0.2617 | -7.3% | -24.1% |
| fiqa | BM25 > ce-l6 @25 | 0.3046 | 0.2824 | 0.2259 | -7.3% | -25.9% |
| fiqa | BM25 > ce-l6 @100 | 0.3307 | — | — | — | — |

## ASR word error rate of the spoken queries

Mean / median WER, and the share of transcripts that are hallucinations (no word in common with the query, WER > 1, and character similarity < 0.35 — an invention rather than a sound-alike mishearing; a known Whisper failure on very short noisy clips).

| Dataset | Median query words | ASR wideband | ASR telephone |
|---|---|---|---|
| scifact | 12 | 0.189 / 0.143 (0% halluc.) | 0.386 / 0.333 (0% halluc.) |
| nfcorpus | 2 | 0.321 / 0.000 (0% halluc.) | 0.944 / 0.667 (7% halluc.) |
| fiqa | 10 | 0.060 / 0.000 (0% halluc.) | 0.186 / 0.125 (0% halluc.) |

## Re-ranking depth vs effectiveness vs latency (text queries)

Latency hardware: CPU = x86_64 (4 threads); GPU = Tesla T4. Batch = depth (one forward pass per voice turn), warm model, 40 queries per dataset.

| Dataset | Depth | nDCG@10 | Hit@1 | CE latency CPU p50 / p95 (ms) | CE latency GPU p50 (ms) |
|---|---|---|---|---|---|
| scifact | BM25 only | 0.6656 | 0.5433 | 0 | 0 |
| scifact | 10 | 0.6716 | 0.5533 | 433 / 471 | 36 |
| scifact | 25 | 0.6737 | 0.5633 | 1160 / 1205 | 80 |
| scifact | 50 | 0.6685 | 0.5600 | 2193 / 2280 | 165 |
| scifact | 100 | 0.6653 | 0.5600 | 4760 / 4851 | 339 |
| nfcorpus | BM25 only | 0.3084 | 0.4365 | 0 | 0 |
| nfcorpus | 10 | 0.3208 | 0.4954 | 413 / 447 | 33 |
| nfcorpus | 25 | 0.3307 | 0.4985 | 1144 / 1195 | 79 |
| nfcorpus | 50 | 0.3337 | 0.4985 | 2279 / 2357 | 171 |
| nfcorpus | 100 | 0.3383 | 0.5015 | 5345 / 5463 | 363 |
| fiqa | BM25 only | 0.2328 | 0.2160 | 0 | 0 |
| fiqa | 10 | 0.2675 | 0.2901 | 375 / 387 | 31 |
| fiqa | 25 | 0.3046 | 0.3148 | 1108 / 1146 | 75 |
| fiqa | 50 | 0.3205 | 0.3225 | 2231 / 2280 | 160 |
| fiqa | 100 | 0.3307 | 0.3225 | 5299 / 5399 | 342 |

## Ablations at the production depth (BM25 top-25, text queries)

| Dataset | Reranker | Params | Max tokens | nDCG@10 | Hit@1 | p vs ce-l6 |
|---|---|---|---|---|---|---|
| scifact | ce-l6 | 22.7M | 256 | 0.6737 | 0.5633 | — |
| scifact | ce-l12 | 33.4M | 256 | 0.6872 | 0.5800 | 0.013 |
| scifact | ce-l6-512 | 22.7M | 512 | 0.6785 | 0.5667 | 0.491 |
| nfcorpus | ce-l6 | 22.7M | 256 | 0.3307 | 0.4985 | — |
| nfcorpus | ce-l12 | 33.4M | 256 | 0.3291 | 0.4985 | 0.411 |
| nfcorpus | ce-l6-512 | 22.7M | 512 | 0.3320 | 0.4923 | 0.633 |
| fiqa | ce-l6 | 22.7M | 256 | 0.3046 | 0.3148 | — |
| fiqa | ce-l12 | 33.4M | 256 | 0.3073 | 0.3102 | 0.319 |
| fiqa | ce-l6-512 | 22.7M | 512 | 0.3079 | 0.3133 | 0.364 |

## Score interpolation (BM25 ⊕ CE, depth 25): alpha chosen by 5-fold CV

| Dataset | Condition | alphas picked per fold | nDCG@10 at alpha=0 | best alpha in hindsight |
|---|---|---|---|---|
| scifact | Text (oracle) | 0.3, 0.5, 0.3, 0.5, 0.5 | 0.6737 | 0.5 (0.6993) |
| scifact | ASR wideband | 0.3, 0.2, 0.4, 0.4, 0.4 | 0.5941 | 0.4 (0.6051) |
| scifact | ASR telephone | 0.3, 0.3, 0.3, 0.2, 0.3 | 0.5088 | 0.3 (0.5186) |
| nfcorpus | Text (oracle) | 0.1, 0.0, 0.0, 0.4, 0.1 | 0.3307 | 0.1 (0.3310) |
| nfcorpus | ASR wideband | 0.0, 0.1, 0.0, 0.4, 0.0 | 0.2589 | 0.1 (0.2591) |
| nfcorpus | ASR telephone | 0.1, 0.2, 0.1, 0.4, 0.1 | 0.1574 | 0.1 (0.1579) |
| fiqa | Text (oracle) | 0.1, 0.1, 0.1, 0.1, 0.0 | 0.3046 | 0.1 (0.3059) |
| fiqa | ASR wideband | 0.1, 0.1, 0.1, 0.1, 0.1 | 0.2824 | 0.1 (0.2837) |
| fiqa | ASR telephone | 0.0, 0.1, 0.1, 0.1, 0.1 | 0.2259 | 0.1 (0.2267) |

## Abstention: can the system tell when the knowledge base has no answer?

AUROC separates in-domain answerable queries from out-of-domain queries (no answer exists) using the top score of the BM25 top-25 pool. AURC is the area under the risk–coverage curve for Hit@1 on in-domain queries (lower is better).

| Dataset | AUROC (BM25 score) | AUROC (CE score) | AURC (BM25 conf.) | AURC (CE conf.) | Keep @ 90% OOD rejected | Hit@1 when kept | CE threshold (logit) |
|---|---|---|---|---|---|---|---|
| scifact | 0.958 | 1.000 | 0.273 | 0.246 | 100.0% | 0.563 | -9.48 |
| nfcorpus | 0.565 | 0.998 | 0.413 | 0.295 | 99.3% | 0.544 | -9.38 |
| fiqa | 0.746 | 0.995 | 0.660 | 0.530 | 98.9% | 0.317 | -6.34 |

## Noise-aware fine-tuning (re-rank BM25 top-25 — the production setting)

nDCG@10 is the mean over seeds of per-query scores; ± is the standard deviation of the dataset-level score across seeds (n = seeds). Significance: paired randomization test on seed-averaged per-query nDCG@10, Holm-corrected within protocol × dataset × condition. † p<0.05 vs zero-shot; ‡ p<0.05 vs FT clean (the domain-adaptation control). Under protocol A, a seed whose best epoch was 0 *is* the zero-shot model.


### Protocol A. early stopping on validation


**Text (oracle)**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.6737 | — | 0.5633 | — | — |
| scifact | FT clean (control) | 3 | 0.6893 | 0.0031 | 0.5733 | +0.0156 | — |
| scifact | FT noisy | 3 | 0.6846 | 0.0097 | 0.5767 | +0.0110 | -0.0047 |
| scifact | FT mixed | 3 | 0.6841 | 0.0095 | 0.5722 | +0.0104 | -0.0052 |
| scifact | FT distill (Part 2) | 3 | 0.6833 | 0.0012 | 0.5833 | +0.0096 | -0.0060 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.3307 | — | 0.4985 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.2821† | 0.0043 | 0.3591 | -0.0486 | — |
| nfcorpus | FT noisy | 3 | 0.3082†‡ | 0.0196 | 0.4262 | -0.0225 | +0.0261 |
| nfcorpus | FT mixed | 3 | 0.3034†‡ | 0.0239 | 0.4149 | -0.0273 | +0.0213 |
| nfcorpus | FT distill (Part 2) | 3 | 0.3293‡ | 0.0007 | 0.4964 | -0.0014 | +0.0473 |
| fiqa | zero-shot (MS MARCO) | — | 0.3046 | — | 0.3148 | — | — |
| fiqa | FT clean (control) | 3 | 0.3054 | 0.0030 | 0.3133 | +0.0008 | — |
| fiqa | FT noisy | 3 | 0.3057 | 0.0013 | 0.3117 | +0.0010 | +0.0002 |
| fiqa | FT mixed | 3 | 0.3058 | 0.0011 | 0.3092 | +0.0012 | +0.0004 |
| fiqa | FT distill (Part 2) | 3 | 0.3065 | 0.0018 | 0.3128 | +0.0018 | +0.0010 |

**ASR wideband**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.5941 | — | 0.4900 | — | — |
| scifact | FT clean (control) | 3 | 0.5943 | 0.0055 | 0.4944 | +0.0002 | — |
| scifact | FT noisy | 3 | 0.5979 | 0.0033 | 0.4989 | +0.0038 | +0.0036 |
| scifact | FT mixed | 3 | 0.5960 | 0.0061 | 0.4944 | +0.0020 | +0.0017 |
| scifact | FT distill (Part 2) | 3 | 0.6029 | 0.0012 | 0.5156 | +0.0088 | +0.0086 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.2589 | — | 0.3653 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.2127† | 0.0030 | 0.2611 | -0.0462 | — |
| nfcorpus | FT noisy | 3 | 0.2365†‡ | 0.0195 | 0.3055 | -0.0224 | +0.0238 |
| nfcorpus | FT mixed | 3 | 0.2334†‡ | 0.0227 | 0.3003 | -0.0255 | +0.0207 |
| nfcorpus | FT distill (Part 2) | 3 | 0.2584‡ | 0.0005 | 0.3653 | -0.0005 | +0.0457 |
| fiqa | zero-shot (MS MARCO) | — | 0.2824 | — | 0.3009 | — | — |
| fiqa | FT clean (control) | 3 | 0.2768 | 0.0020 | 0.2850 | -0.0056 | — |
| fiqa | FT noisy | 3 | 0.2795 | 0.0025 | 0.2896 | -0.0029 | +0.0027 |
| fiqa | FT mixed | 3 | 0.2790 | 0.0030 | 0.2865 | -0.0034 | +0.0022 |
| fiqa | FT distill (Part 2) | 3 | 0.2838 | 0.0009 | 0.3014 | +0.0014 | +0.0070 |

**ASR telephone**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.5088 | — | 0.4200 | — | — |
| scifact | FT clean (control) | 3 | 0.4901 | 0.0029 | 0.3889 | -0.0186 | — |
| scifact | FT noisy | 3 | 0.5072 | 0.0054 | 0.4100 | -0.0016 | +0.0171 |
| scifact | FT mixed | 3 | 0.5053 | 0.0088 | 0.4067 | -0.0034 | +0.0152 |
| scifact | FT distill (Part 2) | 3 | 0.5160 | 0.0019 | 0.4244 | +0.0072 | +0.0258 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.1574 | — | 0.2291 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.1228† | 0.0019 | 0.1538 | -0.0346 | — |
| nfcorpus | FT noisy | 3 | 0.1387†‡ | 0.0164 | 0.1785 | -0.0188 | +0.0159 |
| nfcorpus | FT mixed | 3 | 0.1371†‡ | 0.0178 | 0.1734 | -0.0204 | +0.0143 |
| nfcorpus | FT distill (Part 2) | 3 | 0.1560‡ | 0.0003 | 0.2270 | -0.0015 | +0.0332 |
| fiqa | zero-shot (MS MARCO) | — | 0.2259 | — | 0.2469 | — | — |
| fiqa | FT clean (control) | 3 | 0.2115† | 0.0045 | 0.2140 | -0.0144 | — |
| fiqa | FT noisy | 3 | 0.2189†‡ | 0.0063 | 0.2274 | -0.0069 | +0.0074 |
| fiqa | FT mixed | 3 | 0.2196†‡ | 0.0065 | 0.2284 | -0.0063 | +0.0081 |
| fiqa | FT distill (Part 2) | 3 | 0.2267‡ | 0.0009 | 0.2443 | +0.0008 | +0.0152 |

**Macro-average over datasets, protocol A (nDCG@10)**

| Reranker | Text (oracle) | ASR wideband | ASR telephone | Text→telephone drop |
|---|---|---|---|---|
| zero-shot (MS MARCO) | 0.4363 | 0.3785 | 0.2974 | -31.9% |
| FT clean (control) | 0.4256 | 0.3613 | 0.2748 | -35.4% |
| FT noisy | 0.4328 | 0.3713 | 0.2883 | -33.4% |
| FT mixed | 0.4311 | 0.3695 | 0.2873 | -33.4% |
| FT distill (Part 2) | 0.4397 | 0.3817 | 0.2995 | -31.9% |

### Protocol B. fixed budget (3 epochs)


**Text (oracle)**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.6737 | — | 0.5633 | — | — |
| scifact | FT clean (control) | 3 | 0.6861 | 0.0042 | 0.5667 | +0.0124 | — |
| scifact | FT noisy | 3 | 0.6857 | 0.0067 | 0.5700 | +0.0120 | -0.0004 |
| scifact | FT mixed | 3 | 0.6891 | 0.0036 | 0.5778 | +0.0154 | +0.0030 |
| scifact | FT distill (Part 2) | 3 | 0.6828 | 0.0021 | 0.5833 | +0.0091 | -0.0033 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.3307 | — | 0.4985 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.2758† | 0.0036 | 0.3426 | -0.0549 | — |
| nfcorpus | FT noisy | 3 | 0.2866†‡ | 0.0076 | 0.3591 | -0.0441 | +0.0108 |
| nfcorpus | FT mixed | 3 | 0.2846†‡ | 0.0075 | 0.3550 | -0.0461 | +0.0088 |
| nfcorpus | FT distill (Part 2) | 3 | 0.3291‡ | 0.0005 | 0.4943 | -0.0016 | +0.0533 |
| fiqa | zero-shot (MS MARCO) | — | 0.3046 | — | 0.3148 | — | — |
| fiqa | FT clean (control) | 3 | 0.3029 | 0.0052 | 0.3056 | -0.0018 | — |
| fiqa | FT noisy | 3 | 0.3015 | 0.0052 | 0.2999 | -0.0032 | -0.0014 |
| fiqa | FT mixed | 3 | 0.3050 | 0.0029 | 0.3066 | +0.0004 | +0.0021 |
| fiqa | FT distill (Part 2) | 3 | 0.3065 | 0.0018 | 0.3128 | +0.0019 | +0.0037 |

**ASR wideband**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.5941 | — | 0.4900 | — | — |
| scifact | FT clean (control) | 3 | 0.5845 | 0.0121 | 0.4778 | -0.0095 | — |
| scifact | FT noisy | 3 | 0.5922 | 0.0070 | 0.4867 | -0.0019 | +0.0077 |
| scifact | FT mixed | 3 | 0.5941 | 0.0076 | 0.4933 | +0.0000 | +0.0096 |
| scifact | FT distill (Part 2) | 3 | 0.6027 | 0.0014 | 0.5144 | +0.0086 | +0.0181 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.2589 | — | 0.3653 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.2079† | 0.0021 | 0.2518 | -0.0510 | — |
| nfcorpus | FT noisy | 3 | 0.2165†‡ | 0.0068 | 0.2601 | -0.0423 | +0.0087 |
| nfcorpus | FT mixed | 3 | 0.2160†‡ | 0.0087 | 0.2580 | -0.0429 | +0.0081 |
| nfcorpus | FT distill (Part 2) | 3 | 0.2586‡ | 0.0008 | 0.3653 | -0.0003 | +0.0507 |
| fiqa | zero-shot (MS MARCO) | — | 0.2824 | — | 0.3009 | — | — |
| fiqa | FT clean (control) | 3 | 0.2738 | 0.0048 | 0.2773 | -0.0086 | — |
| fiqa | FT noisy | 3 | 0.2737 | 0.0062 | 0.2788 | -0.0087 | -0.0001 |
| fiqa | FT mixed | 3 | 0.2763 | 0.0044 | 0.2788 | -0.0061 | +0.0025 |
| fiqa | FT distill (Part 2) | 3 | 0.2841 | 0.0006 | 0.3020 | +0.0017 | +0.0103 |

**ASR telephone**

| Dataset | Reranker | n | nDCG@10 | ± seed sd | Hit@1 | Δ vs zero-shot | Δ vs FT clean |
|---|---|---|---|---|---|---|---|
| scifact | zero-shot (MS MARCO) | — | 0.5088 | — | 0.4200 | — | — |
| scifact | FT clean (control) | 3 | 0.4872 | 0.0121 | 0.3867 | -0.0215 | — |
| scifact | FT noisy | 3 | 0.5035‡ | 0.0071 | 0.4000 | -0.0053 | +0.0163 |
| scifact | FT mixed | 3 | 0.5033‡ | 0.0074 | 0.3978 | -0.0055 | +0.0161 |
| scifact | FT distill (Part 2) | 3 | 0.5146 | 0.0022 | 0.4211 | +0.0058 | +0.0274 |
| nfcorpus | zero-shot (MS MARCO) | — | 0.1574 | — | 0.2291 | — | — |
| nfcorpus | FT clean (control) | 3 | 0.1198† | 0.0012 | 0.1445 | -0.0376 | — |
| nfcorpus | FT noisy | 3 | 0.1240†‡ | 0.0027 | 0.1373 | -0.0334 | +0.0042 |
| nfcorpus | FT mixed | 3 | 0.1251†‡ | 0.0035 | 0.1414 | -0.0323 | +0.0053 |
| nfcorpus | FT distill (Part 2) | 3 | 0.1559‡ | 0.0002 | 0.2250 | -0.0016 | +0.0361 |
| fiqa | zero-shot (MS MARCO) | — | 0.2259 | — | 0.2469 | — | — |
| fiqa | FT clean (control) | 3 | 0.2045† | 0.0100 | 0.2016 | -0.0213 | — |
| fiqa | FT noisy | 3 | 0.2095†‡ | 0.0063 | 0.2078 | -0.0163 | +0.0050 |
| fiqa | FT mixed | 3 | 0.2107†‡ | 0.0052 | 0.2078 | -0.0152 | +0.0061 |
| fiqa | FT distill (Part 2) | 3 | 0.2274‡ | 0.0004 | 0.2459 | +0.0015 | +0.0228 |

**Macro-average over datasets, protocol B (nDCG@10)**

| Reranker | Text (oracle) | ASR wideband | ASR telephone | Text→telephone drop |
|---|---|---|---|---|
| zero-shot (MS MARCO) | 0.4363 | 0.3785 | 0.2974 | -31.9% |
| FT clean (control) | 0.4216 | 0.3554 | 0.2705 | -35.8% |
| FT noisy | 0.4246 | 0.3608 | 0.2790 | -34.3% |
| FT mixed | 0.4262 | 0.3621 | 0.2797 | -34.4% |
| FT distill (Part 2) | 0.4395 | 0.3818 | 0.2993 | -31.9% |

## Seen vs unseen speaker voice (ASR conditions, BM25 top-25, nDCG@10)

| Condition | Voice group | Queries | zero-shot | FT mixed (protocol B, seed mean) | Δ |
|---|---|---|---|---|---|
| ASR wideband | seen voices | 1008 | 0.3464 | 0.3327 | -0.0138 |
| ASR wideband | unseen voice (Moira) | 263 | 0.3637 | 0.3488 | -0.0148 |
| ASR telephone | seen voices | 1008 | 0.2708 | 0.2546 | -0.0162 |
| ASR telephone | unseen voice (Moira) | 263 | 0.2924 | 0.2710 | -0.0214 |
