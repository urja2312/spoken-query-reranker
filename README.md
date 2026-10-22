# Does the Reranker Hear You?

**Robustness of neural retrieval to spoken queries, and noise-aware training of cross-encoder
rerankers for voice agents.**

Mini project for **Deep Learning (702AI0C008)** and **Natural Language Processing (702AI0C009)**,
MPSTME, NMIMS Mumbai, AY 2026–27.

📄 [Research paper](paper/Paper.docx) · 📘 [Project report](paper/Project_Report.docx) ·
📝 [Pre-registration](paper/PREREGISTRATION.md) · 📚 [Literature review](paper/LITERATURE_REVIEW.md)

---

## The problem

A voice agent answers a caller in three steps: speech recognition (ASR) turns the voice into text,
a search system finds relevant passages, and a **reranker** — a transformer cross-encoder — puts the
best passage on top. The reranker is trained on typed text, but in a voice agent it only ever sees
the ASR transcript, and ASR makes mistakes ("kohlrabi" → "Cole Robbie").

We measure how much spoken queries hurt retrieval, and whether deep-learning training methods can
make the reranker robust to them.

```
Caller ──► ASR ──► transcript ──► BM25 / dense / hybrid search ──► Reranker ──► answer
                      ▲                                                ▲
              errors enter here                        only ever sees the transcript
```

## Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | Spoken queries cost **7–25%** of nDCG@10 over a microphone and **24–58%** over a telephone line; the reranker does not reduce the relative loss. | 3 BEIR collections, 5 accents, Whisper |
| 2 | On **real speech** in 11 English dialects the loss is **11%** on average and **17% for Indian English**; for Indian speakers the en-IN recogniser is *worse* than en-US. | SD-QA, 747 questions × 21 conditions |
| 3 | Fine-tuning the reranker on clean text makes it **more fragile** to speech (significant on two models). | MiniLM-L6 and bge-reranker-base, 3 seeds |
| 4 | **Consistency distillation** (label-free: the reranker learns to score a transcript the way the original model scores the clean question) avoids that damage, and on bge-reranker-base **beats the zero-shot model in 7 of 9 settings, never worse**. | Pre-registered replication |
| 5 | Misrecognised **content words** (names, nouns, numbers), not function words, drive the loss; in ~**half** of failures the right passage never reaches the reranker. | Word-aligned error analysis |
| 6 | The biggest single gain on real speech comes from the **first stage**: re-ranking a hybrid (BM25 + dense) list raises nDCG@10 from 0.516 to 0.607. | SD-QA (exploratory; pre-registered replication in progress) |

All experiments use three random seeds, paired randomization tests with Holm correction, and
checkpoint-selection rules fixed before any test result. Part 3 hypotheses were
[pre-registered](paper/PREREGISTRATION.md) and committed before any Part 3 experiment ran.

## Repository layout

```
src/                      Retrieval layer of the voice agent (used unchanged by the benchmark)
  oncall_kb.py            BM25 search over SQLite FTS5, with the production query builder
  oncall_reranker.py      Re-ranking stage: optional cross-encoder with a safe BM25 fallback
benchmark/                Experiments
  bench/                  Data loading, first-stage retrievers, rerankers, speech, metrics, statistics
  run_speech.py           Text → TTS → (telephone channel) → Whisper transcripts
  build_trainset.py       Training queries with hard negatives
  finetune.py             Fine-tuning: clean / noisy / mixed / consistency distillation
  run_bench.py            Evaluation grid (through the production re-ranking stage)
  build_sdqa.py           SD-QA real-speech passage-retrieval benchmark
  part3_data.py           Whisper SD-QA, HeySQuAD, SLUE-SQA-5 and real-vs-synthetic training data
  analyze*.py             Result tables and significance tests
  error_analysis.py       Which words break, and where the loss happens
  make_figures.py         Figures used in the paper
  pipeline*.sh            End-to-end pipelines (Part 1, 2, 3)
  results/                Result tables (Markdown/CSV/JSON) and training logs
kaggle/                   Runner used to execute the pipelines on Kaggle GPUs
paper/                    Paper, project report, pre-registration, literature review, figures
tests/                    Unit tests for the re-ranking stage
```

## Models and data

| Role | Model / dataset |
|---|---|
| Rerankers | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M), `BAAI/bge-reranker-base` (278M), `BAAI/bge-reranker-v2-m3` (568M) |
| Dense retriever | `BAAI/bge-small-en-v1.5`; hybrid by reciprocal rank fusion (k = 60) |
| Speech recognition | Whisper base.en and large-v3-turbo (faster-whisper); Google ASR transcripts from SD-QA |
| Test collections | BEIR SciFact, NFCorpus, FiQA; SD-QA (11 English dialects); HeySQuAD; SLUE-SQA-5 |
| Speech synthesis | macOS voices (5 accents, Part 1); Piper (Part 3) |

Datasets and model weights are downloaded by the scripts and are not stored in this repository.

## Reproducing the results

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests              # 6 tests

cd benchmark
./pipeline.sh                                      # Part 1: BEIR, synthetic speech, fine-tuning
MODE=A ./pipeline_part2.sh; MODE=B ./pipeline_part2.sh   # Part 2: SD-QA, distillation
MODE=C ./pipeline_part3.sh                         # Part 3 data (then MODE=E, H, S)
python -I analyze.py; python -I analyze_sdqa.py; python -I analyze_part3.py
python -I error_analysis.py; python -I make_figures.py
```

Training and large evaluations were run on Kaggle (NVIDIA Tesla T4) with `kaggle/run_on_kaggle.py`.
One fine-tuning run of MiniLM-L6 takes about 5 minutes on a T4; retraining on a fresh machine
reproduced every validation score to four decimal places.

## Status

Parts 1 and 2, and three of the six pre-registered Part 3 tests, are complete. The remaining three
(hybrid first stage on new real-speech test sets, real vs. synthetic training noise, distillation from
a larger teacher) are running.

## Team

[Student Name 1] · [Student Name 2] · [Student Name 3] · [Student Name 4]
Faculty: [Faculty names] — MPSTME, NMIMS Mumbai

## Licence

Code is released under the [MIT Licence](LICENSE). Datasets and pre-trained models keep their own
licences.
