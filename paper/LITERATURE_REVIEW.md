# Paper review, literature review and novelty positioning

*Working title:* **Does the Reranker Hear You? Noise-Aware Fine-Tuning of Cross-Encoder Rerankers for Spoken Queries in Voice Agents**

*Status of this document:* written 11 Oct 2026, while the seed 0 / seeds 1–2 Kaggle runs are in progress. Every
claim about prior work below was checked against the paper itself or its arXiv abstract during
this review; the source is linked in §7. Novelty can never be *proven* — the gap statement in §3
is "to our knowledge, after targeted searches", and must stay worded that way in the paper.

---

## 1. One-paragraph summary of our paper

Voice agents retrieve with a *transcript*, not with what the caller said. We build spoken versions
of three BEIR test collections (SciFact, NFCorpus, FiQA; 1,271 queries) by synthesising each query in
five English accents (incl. Indian English) and transcribing it with Whisper over two channels —
wideband (app/browser mic) and a simulated telephone line (8 kHz, 300–3400 Hz, G.711 μ-law, babble
noise at 10 dB SNR). We measure how a production two-stage pipeline (BM25 → MiniLM cross-encoder),
plus dense and hybrid first stages, degrades from typed to spoken queries, and we test whether
**fine-tuning the cross-encoder on ASR transcripts** buys robustness beyond ordinary in-domain
fine-tuning — using a clean-text fine-tuning control, two checkpoint-selection protocols and three
seeds. We also characterise a failure mode specific to short spoken queries: ASR *hallucination*.

---

## 2. Literature review (grouped; all checked)

### 2.1 Speech errors and retrieval / QA — the closest line of work

| Work | What it does | What it does **not** do (our opening) |
|---|---|---|
| **Sidiropoulos, Vakulenko & Kanoulas, CIKM 2022** — *On the Impact of Speech Recognition Errors in Passage Retrieval for Spoken QA* | Google TTS → wav2vec 2.0 noise on MS MARCO + NQ; accents incl. en-IN; 700 human-spoken NQ questions. BM25 and dense (DPR) MRR@10 drop **62.7% / 57.7%** under ASR noise; ASR-noise data augmentation is the best fix for the **dense retriever**; natural noise hurts more than synthetic. | States explicitly: *"We follow the dual-encoder architecture compared to a cross-encoder one"*. **No reranker.** Open-domain only; no telephone channel; no Whisper. |
| **Sidiropoulos, PhD thesis, U. Amsterdam 2025** | Extends the above: typo-robust dense retrievers, multimodal dense retrieval for spoken queries (speech encoder + text). | ASR work is dual-encoder only; re-ranker robustness is discussed **only for typos**, citing others. |
| **Li et al., SQuTR, arXiv 2602.12783 (2026)** | Spoken-query retrieval benchmark: 37k queries from 6 datasets (incl. **FiQA**), CosyVoice-3 TTS with 200 speaker profiles, environmental + babble noise at 20/10/0 dB SNR; Whisper sizes, 12+ embedding models, BM25, one end-to-end audio model. Finding: the retrieval backend matters more than ASR size. | **No cross-encoder / reranker stage; no fine-tuning of any model; no telephone channel or codec.** |
| **Bao, EMNLP 2026** — *Better Retrieval, Worse Robustness: Multi-hop RAG Amplifies ASR Errors* | 4 TTS accents on HotpotQA / 2Wiki / MuSiQue; graph-linking and iterative RAG amplify ASR errors; 87–96% of failures are corrupted entities. | No reranker training; no telephone; multi-hop RAG, not reranking. |
| **Moriya & Jones, SLT 2022** — *Noise Robustness for Spoken Content Retrieval…* | BERT re-ranking + dense retrieval over **ASR transcripts of spoken documents** (How2); MRR −10–14%; N-best and semi-supervised ASR close half the gap. | Spoken **documents**, not spoken **queries**; no noise-aware reranker training. |
| **Li et al., DUPAR, arXiv 2609.18042 (2026)**; **Kim & Ahn, STeReO, Interspeech 2026** | Speech-native retrieval / speech+text reranking orchestration as alternatives to the ASR cascade. | Different paradigm (audio-native). We study the cascade that deployed voice agents actually run — a useful contrast in Related Work. |
| Lee, Wu & Lee, **Spoken SQuAD**, Interspeech 2018; Ravichander et al., **NoiseQA**, EACL 2021; Faisal et al., **SD-QA**, 2021 | ASR noise devastates reading comprehension; accent/dialect matters. | Reading comprehension, not passage ranking. Cite as motivation. |

### 2.2 Robustness of neural rankers to query noise (text side)

| Work | Relevance |
|---|---|
| **Zhuang & Zuccon, EMNLP 2021** — *Dealing with Typos for BERT-based Passage Retrieval and Ranking* | **Typos-aware training of a BERT re-ranker** (and dense retriever) on MS MARCO: robust to typos with no loss on clean queries. **The direct text-side analogue of our method — and the key distinction: typos rarely produce *real words*, ASR errors do** ("reading" → "retting"; "prunes" → whole invented sentence). |
| Zhuang & Zuccon, SIGIR 2022 — CharacterBERT + self-teaching | Typo robustness for dense retrievers; attributes fragility to WordPiece tokenisation. |
| **Ma, dos Santos & Arnold, Findings ACL 2021** — *Contrastive Fine-tuning Improves Robustness for Neural Rankers* | Robust ranker training against query perturbations (text). |
| **Penha, Câmara & Hauff, ECIR 2022** — *Query Variation Generators* | Rerank pipelines lose ~20% under meaning-preserving query variations; argues collections should include variations. Text variations, not channel noise. |
| **Hagen, Scells & Potthast, Findings EMNLP 2024** — *Revisiting Query Variation Robustness* | Reproduces cross-encoder fragility; also holds, less strongly, for LLM bi-encoders. |

### 2.3 Re-ranking, zero-shot transfer and our baselines

- Nogueira & Cho 2019 (BERT passage re-ranking) — the cross-encoder paradigm.
- **Thakur et al., BEIR, NeurIPS 2021 Datasets** — our collections and protocol; our BM25 reproduces the published SciFact/FiQA numbers (0.666 vs 0.665; 0.233 vs 0.236).
- **Rosa et al., arXiv 2212.06121 (2022)** — *In Defense of Cross-Encoders for Zero-Shot Retrieval*: small cross-encoders generalise poorly out of domain; size matters most OOD. **Explains our SciFact result** (MiniLM-L6 reranker gives no significant gain).
- Cormack et al. 2009 (reciprocal rank fusion) — our hybrid; BGE (Xiao et al. 2023) — our dense model.

### 2.4 Speech recognition

- Radford et al., Whisper (ICML 2023) — our ASR.
- **Koenecke et al., *Careless Whisper*, arXiv 2402.08021 (FAccT 2024)** — ~1% of Whisper transcripts contain hallucinated content, more with long non-vocal durations. **They did not isolate query length;** our NFCorpus observation (7% hallucination on 1–2-word telephone queries) is a retrieval-specific extension worth reporting carefully.

---

## 3. The gap — and what we can honestly claim

**What prior work covers:** ASR-noise robustness of *first-stage* retrievers (BM25, dual encoders,
embedding models — Sidiropoulos 2022; SQuTR 2026), of *multi-hop RAG* (Bao 2026), and of rankers over
*spoken documents* (Moriya & Jones 2022). Noise-aware training of *cross-encoder re-rankers* exists, but
**only for typos** (Zhuang & Zuccon 2021; Ma et al. 2021).

**Gap (to our knowledge, after the searches in §7):**

1. **No study fine-tunes a cross-encoder re-ranker on ASR-transcribed queries.** Sidiropoulos did
   this for dual encoders and deliberately excluded cross-encoders; SQuTR trains nothing.
2. **No study separates "noise exposure" from "domain adaptation"** with a clean fine-tuning control
   at equal data and budget. Without that control, any gain from fine-tuning a general MS MARCO
   reranker on in-domain spoken data is uninterpretable — the gain could be the domain, not the
   noise.
3. **No spoken-query retrieval study simulates the telephone channel** (narrowband + μ-law codec +
   babble), which is where call-centre voice agents operate.
4. **Specialist domains** (science, nutrition/medical, finance) are new for the
   ASR → retrieve → re-rank cascade (SQuTR has FiQA but no re-ranking).
5. **Short-query ASR hallucination** as a retrieval failure mode, and **re-rank depth vs latency** for
   a voice-turn budget, are not analysed in this setting.

**Claim strength — be honest:** this is an *empirical analysis paper* with a controlled training
experiment, not a new architecture. Its value is the clean experimental design and the answer to a
practical question. A **negative** result for H2 is still a contribution *because* of gap 1 + 2:
it would show that what worked for dual encoders and for typos does **not** transfer to cross-encoder
re-ranking of ASR queries — and explain why (errors produce real words / entire fabricated
utterances, which no re-ranker can invert).

---

## 4. Final hypotheses (revised 11 Oct 2026, before any test-set result)

*Superseded first draft:* H1 degradation / H2 naive noisy fine-tuning beats clean / H3 first stage
matters more / H4 short-query loss. Revised once real human-speech datasets were found (§4b) and seed-0
validation showed naive noisy fine-tuning gives no gain.

- **H1 — Where the loss happens.** ASR errors degrade the cascade at every stage, but **most of the
  end-to-end loss is a first-stage recall failure** — the relevant passage never reaches the
  reranker — so first-stage improvements (dense/hybrid) recover more than re-ranking does.
  *(Grounded in SQuTR: backend matters more than ASR size; Sidiropoulos 2022: −58–63% first stage.)*
- **H2 — Who gets hurt.** Degradation is largest for **Indian-English and other non-US dialects** and
  for the **telephone channel**, and is driven by **corrupted content words / entities**, not
  function words. *(SD-QA: dialect gaps; Bao 2026: entity corruption is 87–96% of failures.)*
- **H3 — Method (main claim).** **Consistency distillation** (train the cross-encoder so its scores
  for the noisy transcript match its scores for the clean question) improves spoken-query re-ranking
  over clean fine-tuning, **whereas naive training on noisy transcripts with the original labels does
  not** — because some transcripts no longer express the question, so their labels become noise.
  *(Self-teaching worked for typos — Zhuang & Zuccon 2022; never tried for cross-encoders on ASR.)*
- **H4 — Real vs synthetic training noise.** Training on **real human speech errors** (HeySQuAD-human)
  transfers to real accented speech (SD-QA India-North/South) **better** than training on synthetic
  TTS errors. *(Sidiropoulos 2022: natural noise is harder than synthetic.)*
- **H5 (secondary) — Knowing when not to answer.** Very short spoken queries fail through ASR
  hallucination; the reranker's top score detects these (abstains) better than the BM25 score.

## 4b. Data plan (verified availability, 11 Oct 2026)

| Role | Dataset | Speech | Labels for retrieval | Licence | Use |
|---|---|---|---|---|---|
| Specialist domains | BEIR SciFact / NFCorpus / FiQA (ours, existing) | **Synthetic** TTS, 5 accents, wideband + phone | BEIR qrels | per BEIR | H1, H2, H5 |
| **Real accents (test)** | **SD-QA English** (Faisal et al., Findings EMNLP 2021) — HF `WillHeld/SD-QA` | **Real humans, 11 dialects incl. India-South (4 spk) & India-North (5 spk)**; same questions in every dialect: dev 1,000 / test 1,031 | TyDi-QA English passage annotations → pooled Wikipedia candidate-passage corpus (to build) | Apache 2.0 | H1, H2, H4 — the controlled accent comparison |
| **Real-speech training** | **HeySQuAD-human** (Wu et al. 2023) — HF `yijingwu/HeySQuAD_human` | Real humans (12 native speakers, no accents): train 71,990 / val 4,158 | SQuAD context = the relevant passage (corpus = all SQuAD contexts) | CC BY 4.0 | H3, H4 training data |
| ASR realism check | **Svarah** (AI4Bharat, Interspeech 2023) — HF `ai4bharat/Svarah` (gated, auto-approve) | Real Indian-accented English, 117 speakers, 65 districts | none (ASR only) | CC BY 4.0 (AIKosh listing) | Show our Whisper WER on real Indian speech vs on our TTS voices |
| Optional natural phone set | Team recordings (~200 FiQA queries over real phone calls) | Real Indian English, real telephone | BEIR FiQA qrels | ours | Real-channel sanity check |

**Searched and rejected:** Spoken SQuAD (TTS only); SQuAD-SRC (6 accents, release/licence unclear);
SLUE-SQA-5 (human questions but licence unconfirmed); Sidiropoulos natural NQ (700 qs, on request
only); AxonData / CallCenterEN call-centre data (real 8 kHz calls, but CC BY-NC and **no retrieval
labels**; CallCenterEN releases transcripts only); AI4Bharat IndicVoices/Lahaja (Indian *languages*,
not English queries); Kaggle searches (no spoken-query retrieval sets); YouTube (no relevance labels,
and Terms of Service/copyright forbid bulk re-use).

---

## 5. Reviewer-style critique of our own paper (and fixes)

| # | Weakness a reviewer will raise | Severity | Fix / status |
|---|---|---|---|
| 1 | **All speech is synthetic (TTS).** Sidiropoulos showed natural ASR noise hurts *more* than synthetic. | **High** | **Record a small natural test set**: 4 team members × ~50 FiQA queries each, read aloud on a phone call (real Indian-English accents, real telephone line). ~200 queries, one afternoon. Report it as a separate "natural" condition. This is the single most valuable addition. |
| 2 | Single random seed. | High | **In progress:** 3 seeds on Kaggle; mean ± sd across seeds, tests on seed-averaged per-query scores. |
| 3 | Early stopping on a 120-query validation set picks among near-ties. | Medium | Done: report protocol A (early stopping) **and** B (fixed 3 epochs), both fixed before test results. |
| 4 | One ASR model (Whisper base.en). SQuTR shows ASR size matters less than retriever, but reviewers will still ask. | Medium | Add Whisper small/large-v3 transcripts for the **test** queries only (Kaggle GPU, minutes). |
| 5 | Small training set (1,184 queries) and pointwise BCE only. Sidiropoulos found 400 noisy queries already help a dual encoder. | Medium | Report it as a finding if H2 fails ("unlike dual encoders, cross-encoders gain nothing even with ~1.2k noisy queries"). Optional: a training-size curve (400 / 800 / 1,184). |
| 6 | Is the MiniLM-L6 reranker too weak? (Rosa et al.: small cross-encoders transfer poorly.) | Medium | Ablation with L12 is queued; state the deployment constraint (CPU latency budget) as the reason for small models. |
| 7 | Query-length confound (NFCorpus queries are 1–2 words). | Medium | Break results down by query length bucket and by WER bucket; report hallucinated vs non-hallucinated queries separately. |
| 8 | Telephone simulation is not real telephony (no packet loss, no real codec chain). | Low–Med | State as limitation; the natural phone recordings (fix 1) partially address it. |
| 9 | Overlap with SQuTR on FiQA. | Low | Position as complementary: they study first-stage embedders under environmental noise; we study re-ranking + telephone + training. Cite them prominently. |
| 10 | N-best ASR hypotheses unused (Moriya & Jones show N-best helps). | Low | Future work. |

---

## 6. Proposed paper structure (8 pages, ACL/ECIR short-paper style)

1. **Introduction** — voice agents retrieve with transcripts; rerankers are trained on typed text; RQ1–RQ3.
2. **Related work** — §2.1–2.4 condensed; end with the gap (§3).
3. **Spoken query construction** — TTS voices (5 accents, one held out), channels, Whisper, WER table, hallucination definition.
4. **Retrieval pipeline** — BM25 (production FTS5), dense, hybrid RRF, cross-encoder via the production seam; metrics; significance protocol.
5. **Noise-aware fine-tuning** — data (train split only, leakage checks), three variants, two protocols, three seeds, hyper-parameters.
6. **Results** — (a) typed baselines; (b) robustness typed→wideband→telephone; (c) fine-tuning table (‡ column is the answer to H2); (d) seen vs held-out voice; (e) depth/latency; (f) abstention.
7. **Analysis** — failure cases (hallucinations, entity corruption), query-length breakdown.
8. **Limitations, ethics** (synthetic voices, accent coverage, no real caller data), **conclusion**.

**Suitable venues (student-feasible):** FIRE (Forum for Information Retrieval Evaluation — an Indian
IR venue, a natural fit for Indian-English voice agents), ACL/AACL Student Research Workshops, ECIR /
SIGIR short-paper or workshop tracks. Check each call for page limits and deadlines.

**Course mapping:** DL — fine-tuning a transformer with AdamW, warm-up/decay, gradient clipping,
dropout, early stopping, a controlled ablation, multi-seed evaluation (CO2). NLP — tokenisation and
BM25, dense embeddings, cross-encoders, ASR error analysis (CO2–CO4).

---

## 7. Sources checked during this review

- Sidiropoulos, Vakulenko, Kanoulas (CIKM 2022): https://arxiv.org/pdf/2209.12944 · https://dl.acm.org/doi/10.1145/3511808.3557662
- Sidiropoulos PhD thesis (2025): https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/georgios-sidiropoulos-phd-thesis-2025.pdf
- SQuTR (2026): https://arxiv.org/abs/2602.12783
- Bao (EMNLP 2026): https://arxiv.org/abs/2608.22872
- Moriya & Jones (SLT 2022): https://arxiv.org/abs/2301.06056
- DUPAR (2026): https://arxiv.org/abs/2609.18042 · STeReO (Interspeech 2026): https://arxiv.org/abs/2608.26194
- Zhuang & Zuccon (EMNLP 2021): https://aclanthology.org/2021.emnlp-main.225 · CharacterBERT+ST: https://deepai.org/publication/characterbert-and-self-teaching-for-improving-the-robustness-of-dense-retrievers-on-queries-with-typos
- Penha, Câmara, Hauff (ECIR 2022): https://arxiv.org/abs/2111.13057v3
- Hagen, Scells, Potthast (Findings EMNLP 2024): https://preview.aclanthology.org/new-sigs/2024.findings-emnlp.248/
- Ma, dos Santos, Arnold (Findings ACL 2021): cited as ref. [96] in the Sidiropoulos thesis
- Rosa et al. (2022): https://arxiv.org/abs/2212.06121v1
- Koenecke et al., Careless Whisper: https://arxiv.org/abs/2402.08021v1
- Spoken SQuAD (Interspeech 2018): https://www.semanticscholar.org/paper/Spoken-SQuAD:-A-Study-of-Mitigating-the-Impact-of-Lee-Wu/efc6c31d98a2f724fec95a9bb532741e8469248e

*Standard references used but not re-fetched in this review* (BEIR, Nogueira & Cho, Whisper, RRF,
BGE, MiniLM, Smucker et al. significance testing): confirm bibliographic details on DBLP before
submission.

## 5. Literature update for Part 3 (searched 21 Oct 2026)

No 2025–2026 work was found that fine-tunes or distils a cross-encoder reranker on ASR errors,
or that evaluates rerankers on real dialectal spoken queries. New related work to cite:

- Y. Chen et al., "VoiceBench: Benchmarking LLM-based voice assistants," TACL 2025 — uses SD-QA;
  Indian English degrades every voice assistant, more on real than on synthetic accents.
- "Out of Style: RAG's fragility to linguistic variation," arXiv:2504.08231, 2025 — re-ranking
  helps most on rewritten queries (text variation, not speech).
- DUPAR, arXiv:2609.18042, 2026 — conversational spoken retrieval; notes ASR error propagation
  in cascaded transcript-based pipelines.
- S. Shon et al., "SLUE Phase-2," ACL 2023 — SLUE-SQA-5 real-speech questions (Part 3 test set).
- C.-J. Lin et al., "SpeechDPR," ICASSP 2024 — end-to-end spoken passage retrieval on SLUE-SQA-5.
- Y. Wu et al., "HeySQuAD," arXiv:2304.13689 — Part 3 real-speech training data.
- J. Chen et al., "BGE M3-Embedding," Findings of ACL 2024 (bge-reranker-v2-m3) — Part 3 modern reranker.
- T. Javed et al., "Svarah," Interspeech 2023 — Indian-accented English ASR benchmark.
