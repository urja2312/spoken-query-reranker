# Part 3 pre-registration — written 21 Oct 2026, before any Part 3 experiment was run

This document fixes the hypotheses, data, models, tests and decision rules for Part 3 **before**
any Part 3 result exists. It is committed to git and uploaded inside the Kaggle bundle (both
timestamped) before the first Part 3 Kaggle job starts. Anything not listed here is reported as
**exploratory**.

## What we had already seen (so it cannot be confirmatory)

- Part 1 BEIR results, including Hybrid → CE on the spoken BEIR conditions.
- Part 2 SD-QA results with Google ASR transcripts, including the exploratory finding that
  Hybrid → CE beats BM25 → CE (0.607 vs 0.516 nDCG@10 on real speech).

Therefore every confirmatory test below uses **data no model has been evaluated on yet**:

| New test set | Speech | ASR | Size |
|---|---|---|---|
| SD-QA-Whisper | real, 11 English dialects (same recordings as Part 2) | Whisper base.en (new transcripts) | 747 questions × 11 dialects |
| HeySQuAD-dev | real, crowd speakers | Whisper base.en | all answerable dev questions |
| SLUE-SQA-5 test | real speakers | Whisper base.en | 2,382 questions (all answerable) |

Whisper base.en is the Part 1 recogniser, so Part 1 and Part 3 share one ASR model (fixes the
Part 1 / Part 2 ASR mismatch). Whisper large-v3-turbo transcripts of SD-QA are added as an
ASR-strength sensitivity analysis (exploratory).

## Confirmatory hypotheses

All tests: two-sided paired randomization test on per-query nDCG@10 (10,000 permutations).
Fine-tuned models: protocol B (fixed 3 epochs), per-query scores averaged over seeds 0, 1, 2.
Multi-condition test sets (SD-QA-Whisper): per-query score averaged over the conditions first.

**C1 — First stage (replicates the Part 2 exploratory result).**
Hybrid (BM25 + bge-small, RRF k = 60) top-25 → ce-l6 has higher nDCG@10 than BM25 top-25 → ce-l6
on each of the three new test sets. Holm correction over the 3 tests, α = 0.05.
*Confirmed* if all three are significant in the predicted direction.

**C2 — Fine-tuning fragility generalises to a modern reranker.**
For BAAI/bge-reranker-base fine-tuned exactly as in Part 1 (same 1,184 BEIR training queries,
same hyper-parameters except fp16 mixed precision), FT-clean has a larger macro typed → telephone
relative nDCG@10 drop than zero-shot bge-reranker-base; FT-distill is not significantly worse than
zero-shot bge-reranker-base in any of the 9 dataset × condition cells (Holm within each cell, as
in Part 1). The macro-drop comparison is reported with a 95% bootstrap CI over queries.

**C3 — Real vs synthetic training noise (H4).**
Training data: 2,000 HeySQuAD **train** questions (+200 validation), each in three forms — typed
text; the real human recording transcribed by Whisper base.en (*real*); the same text synthesised
by open-source TTS and transcribed by Whisper base.en (*synth*). Hard negatives are mined by BM25
over the SQuAD context corpus with the trained form. Models: ce-l6 student, 3 seeds.
Prediction: FT-noisy-real > FT-noisy-synth and FT-distill-real > FT-distill-synth on
(a) SD-QA-Whisper all dialects, (b) SD-QA-Whisper Indian (India-North + India-South),
(c) SLUE-SQA-5 test. Holm over these 6 tests.

**C4 — Which words break (H2).**
Per (query, condition): Δ = nDCG@10(spoken) − nDCG@10(typed) for BM25 → ce-l6. Reference words are
aligned to the transcript (Levenshtein alignment) and tagged with spaCy en_core_web_sm.
Content words: NOUN, PROPN, NUM, ADJ, VERB (non-auxiliary); function words: all others.
OLS: Δ ~ content-word error rate + function-word error rate. Prediction: the content coefficient
is more negative than the function coefficient; 95% CI of the difference from a query-cluster
bootstrap (2,000 resamples) excludes 0. Data: SD-QA-Whisper, SD-QA Google ASR, BEIR spoken.
Secondary: named-entity token error (any entity token misrecognised) as an indicator.

**C5 — Where the loss happens (H1 decomposition).**
Among (query, condition) pairs where BM25 → ce-l6 loses nDCG@10 vs typed, the share in which no
relevant passage remains in the BM25 top-25 pool ("recall failure"), with a 95% bootstrap CI.
Prediction: > 50% on SD-QA-Whisper and on BEIR telephone.

**C6 — Stronger consistency distillation.**
Fixed candidate grid, chosen **on validation only** (BEIR 120 val queries + 200 HeySQuAD val
questions, mean of clean and noisy nDCG@10), seed 0:
- D1: teacher BAAI/bge-reranker-v2-m3, BM25 candidates, τ = 1;
- D2: teacher bge-reranker-v2-m3, hybrid candidates, τ ∈ {1, 2, 4};
- D3: the best of D2 trained on BEIR + HeySQuAD-real training data.
The single best configuration is retrained with seeds 0–2 and tested once against zero-shot ce-l6
on (a) BEIR spoken (macro over 3 datasets × 2 conditions), (b) SD-QA-Whisper, (c) SLUE-SQA-5 test.
Prediction: higher nDCG@10 than zero-shot on all three; Holm over 3 tests.

## Secondary / descriptive (pre-declared, no confirmatory claim)

- Zero-shot bge-reranker-base and bge-reranker-v2-m3 on every test set and condition, and their
  typed → spoken relative drop vs BM25's (does a stronger reranker protect?).
- Whisper base.en vs large-v3-turbo on SD-QA: WER, nDCG@10 loss, Indian-English gap.
- WER of real vs synthetic training transcripts.

## Data hygiene (fixed now, applied before any model is evaluated)

- HeySQuAD/SLUE recordings whose Whisper transcript fails the Part 1 hallucination rule against
  the question text (no shared word, longer than the question, character similarity < 0.35) are
  treated as misaligned audio and dropped from the test sets and from training, for every
  model. The number dropped is reported.
- HeySQuAD: only answerable questions (is_impossible = false). Test = the dev split; training
  questions come from the train split and any training question whose normalised text equals a
  test question is dropped. Training speech is wideband (no telephone channel) for both *real*
  and *synth*, so the two differ only in natural vs synthetic speech.

## Things we will not do

- No test-set result is used to pick a model, a temperature, a depth or a candidate source.
- No hypothesis above is changed after Part 3 results arrive; changes become new, labelled
  exploratory analyses.
