# Part 3 results

Pre-registered in paper/PREREGISTRATION_PART3.md. nDCG@10; re-ranking depth 25; fine-tuned models: protocol B, per-query scores averaged over seeds 0–2; Part 3 scoring in fp16.


## C1. First stage: Hybrid → ce-l6 vs BM25 → ce-l6 on new real-speech test sets (confirmatory)

| Test set | Queries | BM25 | Dense | BM25 → ce-l6 | Hybrid → ce-l6 | Δ | p | p (Holm, 3 tests) |
|---|---|---|---|---|---|---|---|---|
| SD-QA-Whisper (11 dialects) | 704 | 0.2860 | 0.4888 | 0.4691 | 0.5538 | +0.0848 | <0.001 | <0.001 |
| HeySQuAD-dev | 1002 | 0.8526 | 0.7403 | 0.8956 | 0.9059 | +0.0103 | 0.023 | 0.023 |
| SLUE-SQA-5 test | 2382 | 0.3324 | 0.3925 | 0.4690 | 0.5146 | +0.0456 | <0.001 | <0.001 |

## Zero-shot rerankers on typed and spoken queries (descriptive)

Relative change typed → spoken. A reranker that protects against ASR errors would lose a smaller share than BM25.

| Test set | System | Typed | Spoken | Change |
|---|---|---|---|---|
| SD-QA-Whisper (11 dialects) | BM25 | 0.3729 | 0.2860 | -23.3% |
| SD-QA-Whisper (11 dialects) | Hybrid | 0.5280 | 0.4305 | -18.5% |
| SD-QA-Whisper (11 dialects) | BM25 → ce-l6 | 0.5821 | 0.4691 | -19.4% |
| SD-QA-Whisper (11 dialects) | BM25 → bge-base | 0.5528 | 0.4326 | -21.7% |
| SD-QA-Whisper (11 dialects) | BM25 → bge-m3 | 0.6306 | 0.4981 | -21.0% |
| SD-QA-Whisper (11 dialects) | Hybrid → bge-m3 | 0.7476 | 0.6125 | -18.1% |
| HeySQuAD-dev | BM25 | 0.9019 | 0.8526 | -5.5% |
| HeySQuAD-dev | Hybrid | 0.8893 | 0.8473 | -4.7% |
| HeySQuAD-dev | BM25 → ce-l6 | 0.9326 | 0.8956 | -4.0% |
| HeySQuAD-dev | BM25 → bge-base | 0.7915 | 0.7520 | -5.0% |
| HeySQuAD-dev | BM25 → bge-m3 | 0.9456 | 0.9026 | -4.6% |
| HeySQuAD-dev | Hybrid → bge-m3 | 0.9510 | 0.9145 | -3.8% |
| SLUE-SQA-5 test | BM25 | 0.3760 | 0.3324 | -11.6% |
| SLUE-SQA-5 test | Hybrid | 0.5211 | 0.4568 | -12.3% |
| SLUE-SQA-5 test | BM25 → ce-l6 | 0.5209 | 0.4690 | -10.0% |
| SLUE-SQA-5 test | BM25 → bge-base | 0.4145 | 0.3532 | -14.8% |
| SLUE-SQA-5 test | BM25 → bge-m3 | 0.5026 | 0.4607 | -8.3% |
| SLUE-SQA-5 test | Hybrid → bge-m3 | 0.5317 | 0.5006 | -5.8% |
| BEIR clean (macro) | BM25 | 0.4023 | 0.3381 | -15.9% |
| BEIR clean (macro) | Hybrid | 0.4675 | 0.4025 | -13.9% |
| BEIR clean (macro) | BM25 → ce-l6 | 0.4364 | 0.3786 | -13.2% |
| BEIR clean (macro) | BM25 → bge-base | 0.4201 | 0.3653 | -13.0% |
| BEIR clean (macro) | BM25 → bge-m3 | 0.4531 | 0.3926 | -13.4% |
| BEIR clean (macro) | Hybrid → bge-m3 | 0.4967 | 0.4444 | -10.5% |
| BEIR phone (macro) | BM25 | 0.4023 | 0.2536 | -37.0% |
| BEIR phone (macro) | Hybrid | 0.4675 | 0.3206 | -31.4% |
| BEIR phone (macro) | BM25 → ce-l6 | 0.4364 | 0.2973 | -31.9% |
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
| SD-QA-Whisper (11 dialects) | 0.4691 | 0.4519 | 0.4551 | 0.4590 | 0.4727 | 0.4753 |
| SD-QA-Whisper, Indian | 0.4483 | 0.4337 | 0.4356 | 0.4405 | 0.4512 | 0.4532 |
| HeySQuAD-dev | 0.8956 | 0.9064 | 0.9043 | 0.9080 | 0.9019 | 0.9036 |
| SLUE-SQA-5 test | 0.4690 | 0.4504 | 0.3848 | 0.4469 | 0.4214 | 0.4670 |
| BEIR telephone (pooled) | 0.2752 | 0.2656 | 0.2599 | 0.2631 | 0.2719 | 0.2739 |

| Comparison | Δ | p | p (Holm, 6 tests) |
|---|---|---|---|
| SD-QA-Whisper (11 dialects): real vs synth | -0.0039 | 0.067 | 0.202 |
| SD-QA-Whisper (11 dialects): dreal vs dsynth | -0.0025 | 0.090 | 0.202 |
| SD-QA-Whisper, Indian: real vs synth | -0.0049 | 0.023 | 0.090 |
| SD-QA-Whisper, Indian: dreal vs dsynth | -0.0019 | 0.276 | 0.276 |
| SLUE-SQA-5 test: real vs synth | -0.0621 | <0.001 | <0.001 |
| SLUE-SQA-5 test: dreal vs dsynth | -0.0456 | <0.001 | <0.001 |

Training transcripts: mean WER real 0.150, synthetic 0.075 (2181 questions).


## C6. Stronger consistency distillation (confirmatory)

Selected on validation: **ft-d2t4-s0**.

| Candidate | beir/clean | beir/noisy | heysquad/clean | heysquad/noisy | mean |
|---|---|---|---|---|---|
| ft-d1-s0 | 0.4661 | 0.3671 | 0.8785 | 0.7820 | 0.6234 |
| ft-d2t4-s0 | 0.4696 | 0.3664 | 0.8793 | 0.7821 | 0.6244 |
| ft-d3-s0 | 0.4669 | 0.3653 | 0.8785 | 0.7860 | 0.6242 |

| Test set | zero-shot ce-l6 | FT-distill (selected) | Δ | p | p (Holm, 3 tests) |
|---|---|---|---|---|---|
| BEIR spoken (pooled, 3 datasets × 2 conditions) | 0.3127 | 0.3209 | +0.0082 | <0.001 | <0.001 |
| SD-QA-Whisper (11 dialects) | 0.4691 | 0.4764 | +0.0073 | 0.002 | 0.004 |
| SLUE-SQA-5 test | 0.4690 | 0.4753 | +0.0063 | 0.004 | 0.004 |

## ASR strength: Whisper base.en vs large-v3-turbo on SD-QA (descriptive)

| Dialect | WER base | WER turbo | BM25 → ce-l6 base | BM25 → ce-l6 turbo |
|---|---|---|---|---|
| US | 0.070 | 0.029 | 0.5348 | 0.5631 |
| UK | 0.151 | 0.055 | 0.4828 | 0.5495 |
| Australia | 0.142 | 0.057 | 0.4848 | 0.5497 |
| Ireland | 0.110 | 0.055 | 0.5094 | 0.5532 |
| New Zealand | 0.117 | 0.052 | 0.5052 | 0.5516 |
| South Africa | 0.129 | 0.053 | 0.5013 | 0.5574 |
| Nigeria | 0.299 | 0.135 | 0.3631 | 0.4765 |
| Kenya | 0.230 | 0.103 | 0.4058 | 0.5123 |
| Philippines | 0.134 | 0.054 | 0.4760 | 0.5404 |
| India-North | 0.149 | 0.065 | 0.4590 | 0.5353 |
| India-South | 0.190 | 0.075 | 0.4376 | 0.5309 |

Typed questions: 0.5821.

