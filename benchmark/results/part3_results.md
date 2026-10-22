# Part 3 results

Pre-registered in paper/PREREGISTRATION_PART3.md. nDCG@10; re-ranking depth 25; fine-tuned models: protocol B, per-query scores averaged over seeds 0–2; Part 3 scoring in fp16.


## C1. First stage: Hybrid → ce-l6 vs BM25 → ce-l6 on new real-speech test sets (confirmatory)

| Test set | Queries | BM25 | Dense | BM25 → ce-l6 | Hybrid → ce-l6 | Δ | p | p (Holm, 3 tests) |
|---|---|---|---|---|---|---|---|---|

## Zero-shot rerankers on typed and spoken queries (descriptive)

Relative change typed → spoken. A reranker that protects against ASR errors would lose a smaller share than BM25.

| Test set | System | Typed | Spoken | Change |
|---|---|---|---|---|
| BEIR clean (macro) | BM25 | 0.4023 | 0.3381 | -15.9% |
| BEIR clean (macro) | Hybrid | 0.4675 | 0.4025 | -13.9% |
| BEIR clean (macro) | BM25 → ce-l6 | 0.4363 | 0.3785 | -13.3% |
| BEIR clean (macro) | BM25 → bge-base | 0.4201 | 0.3653 | -13.0% |
| BEIR clean (macro) | BM25 → bge-m3 | 0.4531 | 0.3926 | -13.4% |
| BEIR clean (macro) | Hybrid → bge-m3 | 0.4967 | 0.4444 | -10.5% |
| BEIR phone (macro) | BM25 | 0.4023 | 0.2536 | -37.0% |
| BEIR phone (macro) | Hybrid | 0.4675 | 0.3206 | -31.4% |
| BEIR phone (macro) | BM25 → ce-l6 | 0.4363 | 0.2974 | -31.9% |
| BEIR phone (macro) | BM25 → bge-base | 0.4201 | 0.2819 | -32.9% |
| BEIR phone (macro) | BM25 → bge-m3 | 0.4531 | 0.3070 | -32.3% |
| BEIR phone (macro) | Hybrid → bge-m3 | 0.4967 | 0.3611 | -27.3% |

## C2. Fine-tuning a modern reranker (bge-reranker-base) on BEIR (confirmatory)

Macro over SciFact, NFCorpus, FiQA. † p < 0.05 vs zero-shot bge-base (Holm over the 3 variants within the cell).

| Variant | Typed | Wideband | Telephone | Typed → telephone | 95% CI of drop |
|---|---|---|---|---|---|
| zero | 0.4201 | 0.3653 | 0.2819 | -32.9% | [-36.8%, -29.0%] |
| clean | 0.4207 | 0.3616 | 0.2707 | -35.7% | [-39.6%, -31.9%] |
| noisy | 0.4128 | 0.3574 | 0.2710 | -34.3% | [-38.1%, -30.5%] |
| distill | 0.4310 | 0.3757 | 0.2951 | -31.5% | [-35.2%, -27.9%] |

Difference in typed → telephone drop vs zero-shot (percentage points; paired bootstrap over queries, 2,000 resamples):

| Variant | Δ drop | 95% CI |
|---|---|---|
| clean | -2.8 | [-4.8, -0.7] |
| noisy | -1.4 | [-3.5, +0.6] |
| distill | +1.4 | [+0.0, +2.7] |

Per cell (nDCG@10):

| Dataset / condition | zero-shot | FT-clean | FT-noisy | FT-distill |
|---|---|---|---|---|
| scifact / text | 0.6680 | 0.7087† | 0.6925 | 0.6884† |
| scifact / asr_clean | 0.5868 | 0.6110 | 0.6053 | 0.6076† |
| scifact / asr_phone | 0.4913 | 0.4975 | 0.5082 | 0.5187† |
| nfcorpus / text | 0.3037 | 0.2685† | 0.2714† | 0.3098† |
| nfcorpus / asr_clean | 0.2392 | 0.2126† | 0.2098† | 0.2441 |
| nfcorpus / asr_phone | 0.1407 | 0.1172† | 0.1137† | 0.1471† |
| fiqa / text | 0.2885 | 0.2850 | 0.2744† | 0.2948† |
| fiqa / asr_clean | 0.2699 | 0.2612 | 0.2571 | 0.2755 |
| fiqa / asr_phone | 0.2136 | 0.1973† | 0.1911† | 0.2196† |

Variants significantly worse than zero-shot: {'clean': ['nfcorpus/text', 'nfcorpus/asr_clean', 'nfcorpus/asr_phone', 'fiqa/asr_phone'], 'noisy': ['nfcorpus/text', 'nfcorpus/asr_clean', 'nfcorpus/asr_phone', 'fiqa/text', 'fiqa/asr_phone']}


## C3. Real vs synthetic training noise (HeySQuAD-trained ce-l6; confirmatory)

| Test set | zero-shot | FT-clean | FT-noisy-real | FT-noisy-synth | FT-distill-real | FT-distill-synth |
|---|---|---|---|---|---|---|
| BEIR telephone (pooled) | 0.2752 | — | — | — | — | — |

Training transcripts: mean WER real 0.150, synthetic 0.075 (2181 questions).


## C6. Stronger consistency distillation (confirmatory)


## ASR strength: Whisper base.en vs large-v3-turbo on SD-QA (descriptive)

| Dialect | WER base | WER turbo | BM25 → ce-l6 base | BM25 → ce-l6 turbo |
|---|---|---|---|---|
| US | 0.070 | 0.029 | — | — |
| UK | 0.151 | 0.055 | — | — |
| Australia | 0.142 | 0.057 | — | — |
| Ireland | 0.110 | 0.055 | — | — |
| New Zealand | 0.117 | 0.052 | — | — |
| South Africa | 0.129 | 0.053 | — | — |
| Nigeria | 0.299 | 0.135 | — | — |
| Kenya | 0.230 | 0.103 | — | — |
| Philippines | 0.134 | 0.054 | — | — |
| India-North | 0.149 | 0.065 | — | — |
| India-South | 0.190 | 0.075 | — | — |

Typed questions: 0.5821.

