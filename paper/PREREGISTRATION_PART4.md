# Pre-registration — Part 4: fixing recall before the reranker

Committed before any Part 4 experiment was run. Every setting below is fixed in advance; nothing
is tuned on a test set. Any later change is reported as a deviation.

## Motivation (from Parts 1–3)

Pre-registered test C5 found that in 53–68% of the spoken queries that lose quality, the relevant
passage never reaches the reranker. C4 found that misrecognised *content words* (names, nouns,
numbers) drive the loss. Part 3 also showed that training the reranker cannot fix a passage that was
never retrieved. Part 4 therefore tests two training-free methods that act **before** the reranker.
Both attack the content-word errors that the error analysis identified.

**Idea 1 — Listen Twice (LT): retrieval-guided re-transcription.**
1. Whisper transcribes the call (pass 1).
2. Hybrid search returns the top-K passages for that transcript.
3. The rarest content words of those passages become Whisper's decoding prompt (`initial_prompt`).
4. Whisper transcribes the same audio again (pass 2).
5. Search and re-ranking run on the pass-2 transcript.

The knowledge base tells the recogniser which unusual words it should expect.

**Idea 2 — Mishear the Corpus (MC): ASR-simulated document expansion.**
1. Offline, every rare word of the corpus is spoken by a TTS voice and transcribed by the deployment's ASR.
2. If ASR mishears the word, the misheard tokens are added to an expansion field of every passage that
   contains it. For example, "kohlrabi" becomes "cole robbie".
3. A query transcript that contains the same mishearing then matches the passage.

The method adds no query-time latency and needs no training. It never uses test audio or test transcripts.

## Fixed settings

| Setting | Value |
|---|---|
| ASR | Whisper base.en, faster-whisper, GPU float16, greedy, `condition_on_previous_text=False` (the Part 3 pass-1 setting) |
| Rare word | Alphabetic, ≥ 4 letters, not a production stopword, `wordfreq` Zipf frequency < 4.0 |
| LT prompt | K = 5 hybrid passages; N = 30 words, ranked by (passages containing the word) × IDF; comma-separated |
| LT controls | **RND**: same extraction from 5 random non-relevant passages. **ORC**: the relevant passage(s) — an upper bound, not deployable |
| MC vocabulary | Corpus rare words, capped at 40,000 by document frequency |
| MC voices | Piper en_US-lessac-medium and en_GB-alan-medium |
| MC filter | Variant kept only if the transcript differs from the word and fails neither the Part 3 hallucination rule nor the stopword list |
| MC index | Variants go in the unused `section` column of the production FTS5 table; the production query builder is unchanged; the reranker sees the original passage text only |
| Phonetic baseline (PH) | Classic phonetic indexing: Metaphone codes of every word, BM25 over codes, fused with word BM25 by RRF |
| First stage | Hybrid = RRF(lexical, dense bge-small-en-v1.5), k = 60; pool = lexical top-25 ∪ hybrid top-25 |
| Rerankers | ce-l6 (production) for every system; bge-reranker-v2-m3 for base, LT, MC, LT+MC and typed |

## Test sets (real human speech only)

- **SD-QA**: 747 questions × 11 English dialects; the 43 Kenyan questions without audio are dropped everywhere, as in Part 3.
- **HeySQuAD**: 1,002 answerable dev questions.
- **SLUE-SQA-5**: test questions.

The same hygiene rules as Part 3 apply.

## Primary metric and hypotheses

The primary metric is nDCG@10 of **Hybrid → ce-l6 @25**, the best Part 3 pipeline. Scores are compared per
question; for SD-QA each question's score is averaged over its 11 dialects. Each hypothesis is tested
on each of the 3 test sets.

| # | Hypothesis | Comparison |
|---|---|---|
| H1 | Listen Twice improves retrieval | LT > base (pass 1) |
| H2 | The gain comes from retrieval guidance, not from prompting in general | LT > RND |
| H3 | Misheard-corpus expansion improves retrieval | MC > base |
| H4 | ASR-simulated expansion beats classic phonetic indexing | MC > PH |

That gives 12 tests. Each uses a paired randomization test (10,000 sign flips, two-sided, as in Part 3),
with Holm correction over all 12. A hypothesis counts as **supported on a test set** when the difference
is in the predicted direction with Holm-adjusted p < 0.05.

## Secondary analyses (reported, not tested for confirmation)

- **WER and rare-word error rate** of each transcript type. **Prompt injection rate**: words in pass 2 that
  appear in the prompt but in neither the reference nor pass 1.
- **Confirmation-bias check**: the LT effect split by whether pass-1 hybrid top-5 contained a relevant
  passage. We expect gains when it did and possible harm when it did not.
- Share of questions helped or hurt by LT and by MC.
- **LT+MC** (prompt words found by MC-expanded retrieval, pass 2 searched on the MC index) against LT and MC.
- **LT-fuse**: RRF of the pass-1 and pass-2 runs, as a guard against confirmation bias.
- **Typed-query non-inferiority of MC**: margin −0.005 nDCG@10, using the 95% bootstrap CI.
- **bge-reranker-v2-m3 replication**, Recall@25 of the hybrid pool, and the pass-1 re-decode identity check.
- **Cost**: added latency per query (LT) and index growth (MC).
- **HeySQuAD ablation**: LT with K ∈ {1, 3, 10}, labelled as a post-hoc ablation.

## Risks stated in advance

- **LT** may lock in a wrong first retrieval (confirmation bias) and may inject prompt words that were not
  spoken. Both are measured above.
- **MC** may add false matches. Isolated-word TTS lacks the co-articulation of real speech, and the
  vocabulary cap may miss rare words.
