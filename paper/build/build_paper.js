// Builds the mini-project research paper (.docx). All numbers come from results/results.md
// and results/sdqa_results.md (3 seeds, Kaggle T4 runs of 11–16 Oct 2026). Run: node build_paper.js <figdir> <out.docx>
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, Footer, PageNumber, LevelFormat,
} = require("docx");

const [figDir, outPath] = process.argv.slice(2);
const FONT = "Times New Roman";
const W = 9026; // A4 content width in DXA with 1" margins

const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics, superScript: o.sup });
const runs = (spec) => (Array.isArray(spec) ? spec : [spec]).map((x) => (typeof x === "string" ? t(x) : t(x.t, x)));
const P = (spec, o = {}) => new Paragraph({
  children: runs(spec), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: 276 }, indent: o.indent,
  keepNext: o.keepNext,
});
const H1 = (text) => new Paragraph({ children: [t(text, { bold: true, size: 24 })], spacing: { before: 240, after: 120 }, keepNext: true });
const H2 = (text) => new Paragraph({ children: [t(text, { bold: true, italics: true })], spacing: { before: 160, after: 80 }, keepNext: true });
const bullet = (spec) => new Paragraph({ children: runs(spec), numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, alignment: AlignmentType.JUSTIFIED });
const caption = (text) => new Paragraph({ children: [t(text, { size: 19, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 } });
const tcap = (text) => new Paragraph({ children: [t(text, { size: 19, bold: true })], alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true });

function fig(file, wPx, hPx, cap) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(`${figDir}/${file}`), transformation: { width: wPx, height: hPx } })] }),
    caption(cap),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { fill: "E8EEF4", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [t(String(txt), { size: 18, bold: head })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}

const body = [];
const add = (...xs) => body.push(...xs.flat());

// ---------------------------------------------------------------- title block
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [t("Does the Reranker Hear You? Robustness of Neural Retrieval to Spoken Queries and Noise-Aware Fine-Tuning of Cross-Encoder Rerankers for Voice Agents", { size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [t("Arya Shah, Teerth Shah, Urja Singhi", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [t("Mukesh Patel School of Technology Management & Engineering, NMIMS, Mumbai, India", { size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [t("Mini project — Deep Learning (702AI0C008) and Natural Language Processing (702AI0C009), AY 2026–27 · Faculty: Prof. Abhay Kolhe (Deep Learning) · Prof. Soni Sweta and Prof. Manisha Tiwari (Natural Language Processing)", { size: 20, italics: true })] }),
);

add(
  P([{ t: "Abstract— ", bold: true, italics: true }, { t: "Voice agents answer callers by retrieving passages with a speech-recognition (ASR) transcript rather than with what the caller actually said, yet the rerankers in these pipelines are trained on typed text. We measure how a production two-stage pipeline (BM25 followed by a MiniLM cross-encoder reranker), together with dense and hybrid first stages, degrades on spoken queries in two settings: spoken versions of three BEIR collections from science, nutrition/medicine and finance (1,271 queries), synthesised in five English accents and transcribed by Whisper over a wideband channel and a simulated telephone line; and real human speech from SD-QA, 747 Wikipedia questions read aloud by speakers of eleven English dialects including North and South Indian English. Synthetic speech reduces nDCG@10 by 7–25% over wideband audio and 24–58% over the telephone; real speech costs 11.2% on average and 16.9% for Indian English; and the reranker does not reduce the relative loss. Fine-tuning the reranker on clean in-domain text makes it more fragile to speech, and training on transcripts with relevance labels repairs only part of this damage. Label-free consistency distillation, which trains the reranker to score a transcript the way the original model scores the clean question, avoids the damage: it is never significantly worse than the zero-shot model, is significantly better than clean fine-tuning on two of three collections, and is the best reranker on every real-speech group, although its gain over zero-shot (+0.002 to +0.004 nDCG@10) is not significant after correction. The largest improvement on real speech comes from the first stage: re-ranking a hybrid BM25 + dense candidate list instead of a BM25 list raises nDCG@10 from 0.516 to 0.607. In a pre-registered follow-up with a modern 278M-parameter reranker (bge-reranker-base), clean fine-tuning again widened the telephone loss (by 2.8 points, 95% CI 0.7–4.8), while consistency distillation significantly outperformed the zero-shot model in 7 of 9 dataset–condition cells and was never worse. An error analysis shows that misrecognised content words, not function words, drive the loss, and that in more than half of the failures the relevant passage never reaches the reranker. Pre-registered tests on three new real-speech test sets confirm that a hybrid first stage helps (+0.010 to +0.085 nDCG@10) and that distillation from a 568M-parameter teacher significantly improves the 22.7M reranker over its zero-shot self; training on real rather than synthetic speech errors did not help. We discuss implications for voice agents, including the choice of ASR model for Indian-English callers and a re-ranking latency of about 1.1 s on CPU.", italics: true }]),
  P([{ t: "Keywords— ", bold: true, italics: true }, { t: "spoken queries, automatic speech recognition, passage retrieval, cross-encoder reranking, robustness, knowledge distillation, English dialects, voice agents", italics: true }], { after: 200 }),
);

// ---------------------------------------------------------------- I. Introduction
add(H1("I. Introduction"),
  P("Voice agents — telephone helplines, smart speakers and in-app assistants — increasingly answer questions by retrieval-augmented generation: the caller's speech is transcribed by an automatic speech recognition (ASR) model, the transcript is used to retrieve passages from a knowledge base, and an answer is composed from those passages. A retrieval error at this step is unrecoverable: the agent either reads out the wrong instruction or wrongly reports that the information is missing."),
  P("This project grew out of a working bilingual voice support agent that retrieves manual passages with BM25 (SQLite FTS5) and re-ranks them with a cross-encoder. Its retrieval components, like most neural rankers, are trained and evaluated on typed text. Spoken queries differ in two ways that text benchmarks do not capture: ASR errors replace words with other real words (\"reading\" → \"retting\"), and on short or noisy audio the ASR can produce fluent text that was never said. Telephone audio, narrowband and compressed, makes both effects worse."),
  P("We ask four research questions:"),
  bullet([{ t: "RQ1: ", bold: true }, "How much do spoken queries degrade a lexical, dense, hybrid and re-ranked retrieval pipeline, and does the cross-encoder reranker protect against this loss?"]),
  bullet([{ t: "RQ2: ", bold: true }, "Does fine-tuning the reranker on ASR transcripts, or label-free consistency distillation, make it more robust to spoken queries than fine-tuning on clean text, without hurting typed queries?"]),
  bullet([{ t: "RQ3: ", bold: true }, "What do re-ranking depth and model size cost in latency, within the response budget of a live voice turn?"]),
  bullet([{ t: "RQ4: ", bold: true }, "Do these findings hold for real human speech across English dialects, and which speakers and ASR models are most affected?"]),
  P("Our contributions are: (1) spoken versions of three BEIR test collections in specialist domains, under wideband and telephone conditions with five accents; (2) a significance-tested robustness comparison of BM25, dense, hybrid and cross-encoder pipelines on these queries; (3) a controlled fine-tuning study that separates the effect of training on noisy transcripts from the effect of in-domain adaptation, using a clean-text control, three seeds and two pre-declared checkpoint-selection protocols; (4) label-free consistency distillation for cross-encoder rerankers, adapting self-teaching [6] from dense retrievers; (5) a passage-retrieval version of SD-QA [16] that evaluates rerankers on real speech in eleven English dialects; (6) six pre-registered tests on a modern reranker and three new real-speech test sets, including an error analysis of which words break and a negative result for real-speech training; and (7) an effectiveness–latency analysis for deployment.", { before: 60 }),
);

// ---------------------------------------------------------------- II. Related work
add(H1("II. Related Work"),
  H2("A. Speech recognition errors in retrieval and question answering"),
  P("Spoken SQuAD [15] showed that ASR errors severely damage reading comprehension, and SD-QA [16] showed that accuracy varies across English dialects, including Indian English. Closest to our work, Sidiropoulos et al. [1] added synthetic ASR noise (Google TTS and wav2vec 2.0) to MS MARCO and Natural Questions and found that BM25 and dense retrieval lose 63% and 58% of MRR@10; augmenting dense-retriever training with ASR-noised questions was the most effective remedy. They studied dual-encoder retrievers and explicitly did not study cross-encoders. SQuTR [2] is a recent benchmark of 37k spoken queries under environmental noise across six datasets and many embedding models, finding that the retrieval backend matters more than ASR model size; it does not include a reranking stage or any training. Bao [3] shows that multi-hop RAG amplifies ASR errors, with corrupted entities causing most failures, and Moriya and Jones [4] study BERT re-ranking over ASR transcripts of spoken documents rather than spoken queries."),
  H2("B. Robustness of neural rankers to query noise"),
  P("Zhuang and Zuccon [5] showed that BERT re-rankers and dense retrievers fail on queries with typos and proposed typo-aware training, later adding self-teaching for dense retrievers [6]. Ma et al. [7] improved ranker robustness with contrastive fine-tuning, and Penha et al. [8] and Hagen et al. [9] found that retrieval pipelines, including cross-encoders, lose effectiveness under meaning-preserving query variations. These works study textual noise; ASR errors differ in that they usually produce correctly spelled but wrong words."),
  H2("C. Cross-encoders and zero-shot transfer"),
  P("Cross-encoder re-ranking [12] over a BM25 first stage is a standard, strong pipeline. On BEIR [11], small cross-encoders trained on MS MARCO transfer unevenly to specialised domains, and model size matters most out of domain [10]."),
  H2("D. Gap"),
  P("To our knowledge, after targeted searches, no prior study (i) fine-tunes a cross-encoder reranker on ASR-transcribed queries, (ii) separates the benefit of noisy training from in-domain adaptation with a clean-training control, (iii) evaluates spoken queries through a simulated telephone channel on specialist-domain collections, or (iv) tests label-free consistency training for cross-encoders on ASR errors, with an evaluation on real dialectal speech. This study addresses these four points."),
);

// ---------------------------------------------------------------- III. Methodology
add(H1("III. Methodology"),
  H2("A. Retrieval pipeline under study"),
  P("The production pipeline retrieves with BM25 over an SQLite FTS5 index and re-ranks the top-25 candidates with the cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 [19] (22.7M parameters, input truncated to 256 tokens). All benchmark retrieval uses the production query builder and FTS5 engine, and all re-ranking passes through the production re-ranking module, so the measurements describe the deployed code. For comparison we evaluate a dense retriever, BAAI/bge-small-en-v1.5 [18], and a hybrid of BM25 and dense rankings by reciprocal rank fusion (RRF, k = 60) [17]."),
  H2("B. Datasets"),
  P("We use the test splits of three BEIR [11] collections chosen for domain diversity (Table I). Our BM25 reproduces published BEIR baselines (SciFact nDCG@10 0.666 vs. 0.665; FiQA 0.233 vs. 0.236), which validates the evaluation harness."),
  tcap("TABLE I. Test collections"),
  table(["Collection", "Domain", "Documents", "Test queries", "Median query words"],
    [["SciFact", "Scientific claims", "5,183", "300", "12"], ["NFCorpus", "Nutrition / medical", "3,633", "323", "2"], ["FiQA-2018", "Personal finance", "57,638", "648", "10"]],
    [1700, 2200, 1700, 1700, 1726]),
  H2("C. Constructing spoken queries"),
  P("Each test query is synthesised with one of five macOS text-to-speech voices covering five English accents — Indian (Rishi), American (Samantha), British (Daniel), Australian (Karen) and Irish (Moira) — assigned deterministically per query. Audio is transcribed by Whisper base.en [13] (int8, greedy decoding) under two channels:"),
  bullet([{ t: "Wideband: ", bold: true }, "16 kHz audio, as from an app or browser microphone."]),
  bullet([{ t: "Telephone: ", bold: true }, "the same speech mixed with babble from three other queries at 10 dB SNR, down-sampled to 8 kHz, band-pass filtered to 300–3,400 Hz and passed through G.711 µ-law companding."]),
  P("Table II reports word error rates (WER). We define a transcript as a hallucination when it shares no word with the query, is longer than it, and has character similarity below 0.35 (e.g., \"prunes\" → \"I'm going to go to the bathroom.\"). Such transcripts occur only for NFCorpus's one- and two-word queries over the telephone (7%), consistent with reports of Whisper hallucinations [14].", { before: 60 }),
  tcap("TABLE II. ASR word error rate of spoken queries (mean / median)"),
  table(["Collection", "Wideband WER", "Telephone WER", "Telephone hallucinations"],
    [["SciFact", "0.189 / 0.143", "0.386 / 0.333", "0%"], ["NFCorpus", "0.321 / 0.000", "0.944 / 0.667", "7%"], ["FiQA", "0.060 / 0.000", "0.186 / 0.125", "0%"]],
    [2200, 2300, 2300, 2226]),
  H2("D. Noise-aware fine-tuning"),
  P("Training data comes only from each collection's train split: 440 queries per collection (1,320 in total; 1,200 for training and 120 for validation), with test-query identifiers disjoint from training and exact text duplicates of test queries removed. Every training query is spoken by one of four voices; the Irish voice is held out of training entirely so that test results can be split into seen and unseen voices. Each query is transcribed through the wideband or telephone channel (50/50). Sixteen training transcripts classified as hallucinations are removed for every variant, leaving 1,184 training queries. Hard negatives are the top BM25 results for the form of the query being trained on, excluding judged-relevant documents."),
  P("Four variants share identical queries, candidate passages, optimiser and budget. The first three differ only in the form of each training query the model sees:"),
  bullet([{ t: "FT-clean (control): ", bold: true }, "always the original text — isolates the effect of in-domain adaptation."]),
  bullet([{ t: "FT-noisy: ", bold: true }, "always the ASR transcript."]),
  bullet([{ t: "FT-mixed: ", bold: true }, "clean or transcript, re-drawn 50/50 per query per epoch."]),
  bullet([{ t: "FT-distill (consistency distillation): ", bold: true }, "a frozen copy of the zero-shot reranker (the teacher) scores the clean question against its eight candidate passages, and the student is trained so that its score distribution over the same eight passages matches the teacher's, both for the transcript and for the clean question."]),
  P("Starting from the zero-shot checkpoint, each model is trained with pointwise binary cross-entropy over one positive and seven hard negatives per query; AdamW [21] (learning rate 2×10⁻⁵, weight decay 0.01, none on bias and LayerNorm), 10% linear warm-up then linear decay, gradient clipping at 1.0, batch size 32, dropout 0.1, and three epochs (296 steps per epoch). Each run takes 4.5–4.9 minutes on an NVIDIA Tesla T4. FT-distill minimises KL(p_T(·|q) ‖ p_S(·|q̃)) + KL(p_T(·|q) ‖ p_S(·|q)), where q is the clean question, q̃ its transcript, and p the softmax over the eight candidates at temperature τ = 1. It uses no relevance labels in the loss, so unjudged relevant passages among the negatives cannot be pushed down. Each step processes 4 queries × 8 candidates, the same 32 pairs per step and 296 steps per epoch as the other variants (8.7–8.9 minutes per run, since every pair is scored twice). Because a 120-query validation set produces near-ties, we report two checkpoint-selection protocols, both fixed before any test result was seen: (A) early stopping on mean validation nDCG@10 over clean and transcribed forms, and (B) the fixed-budget model after three epochs. Every variant is trained with seeds 0, 1 and 2. Retraining the nine labelled models on a fresh Kaggle instance reproduced every validation score to four decimal places.", { before: 60 }),
  H2("E. Real human speech: SD-QA"),
  P("SD-QA [16] recorded the 1,031 English TyDi-QA development questions read aloud by speakers of eleven English dialects and released transcripts of every recording from Google's speech recogniser, using both a localised model (for example en-IN for India) and the en-US model. We convert it into a passage-retrieval task. The corpus is every candidate passage of every English TyDi-QA development article (46,185 passages from 929 Wikipedia articles); relevant passages are those chosen by TyDi annotators, graded by the number of annotators who chose them; and queries are the 747 questions with at least one annotated passage, joined to TyDi by normalised question text. Each (dialect, ASR model) pair is one test condition, giving 21 conditions (the US recordings have only the en-US transcript). Word error rates range from 0.042 (US) to 0.177 (Kenya, en-US); for India-North and India-South they are 0.102–0.176."),
  P("SD-QA is used only for testing. Every fine-tuned model was trained on the three BEIR collections, so SD-QA measures out-of-domain transfer to real speech. Fine-tuned models are evaluated with protocol B and per-query scores are averaged over the three seeds; all rerankers are scored in half precision (fp16) on a T4 GPU for this benchmark."),
  H2("F. Evaluation"),
  P("We report nDCG@10 (primary), MRR@10, Hit@1 — whether the single passage a voice agent would read aloud is relevant — and Recall@100. Significance uses a two-sided paired randomization test (10,000 permutations) [20] on per-query scores, with Holm correction [24] within each dataset and condition; for the fine-tuned models, per-query scores are first averaged over seeds. Confidence intervals are 95% bootstrap intervals over queries."),
);

// ---------------------------------------------------------------- IV. Results
add(H1("IV. Results"),
  H2("A. Typed-query baselines"),
  P("Table III gives effectiveness on typed queries. The zero-shot reranker significantly improves nDCG@10 over BM25 on NFCorpus and FiQA (p < 0.001; Hit@1 +6.2 and +9.9 points) but not on SciFact (p = 1.0), consistent with poor transfer of small MS MARCO cross-encoders to scientific claims [10]. Improving the first stage is at least as effective: dense retrieval is the best single system on FiQA, and hybrid retrieval followed by re-ranking is best on NFCorpus."),
  tcap("TABLE III. Typed queries: nDCG@10 / Hit@1 (** p < 0.01 vs. BM25, Holm-corrected)"),
  table(["System", "SciFact", "NFCorpus", "FiQA"],
    [["BM25", "0.666 / 0.543", "0.308 / 0.437", "0.233 / 0.216"],
     ["Dense (bge-small)", "0.703 / 0.600", "0.343** / 0.452", "0.391** / 0.390"],
     ["Hybrid (RRF)", "0.706** / 0.587", "0.352** / 0.486", "0.345** / 0.329"],
     ["BM25 → CE @25 (production)", "0.674 / 0.563", "0.331** / 0.499", "0.305** / 0.315"],
     ["BM25 → CE @100", "0.665 / 0.560", "0.338** / 0.502", "0.331** / 0.323"],
     ["Hybrid → CE @25", "0.685 / 0.567", "0.355** / 0.508", "0.369** / 0.341"]],
    [2826, 2066, 2067, 2067]),
  H2("B. Robustness to spoken queries (RQ1)"),
  P("Fig. 1 and Table IV show that every pipeline degrades on spoken queries, mildly over wideband audio and severely over the telephone. The relative loss is similar across BM25, dense, hybrid and re-ranked pipelines: the reranker does not protect against ASR errors, because the errors are already present in the query it receives. NFCorpus, whose queries are one or two words long, loses over half its effectiveness over the telephone, since a single mis-recognised word removes the entire query."),
  ...fig("fig1_robustness.png", 600, 174, "Fig. 1. nDCG@10 for typed, wideband-spoken and telephone-spoken queries."),
  tcap("TABLE IV. Relative nDCG@10 change from typed to spoken queries"),
  table(["System", "SciFact wide / phone", "NFCorpus wide / phone", "FiQA wide / phone"],
    [["BM25", "−14.7% / −31.4%", "−24.7% / −58.1%", "−8.0% / −24.8%"],
     ["Dense (bge-small)", "−12.2% / −26.9%", "−21.2% / −51.9%", "−7.1% / −25.8%"],
     ["Hybrid (RRF)", "−13.6% / −24.5%", "−21.2% / −52.5%", "−7.3% / −24.1%"],
     ["BM25 → CE @25", "−11.8% / −24.5%", "−21.7% / −52.4%", "−7.3% / −25.9%"]],
    [2426, 2200, 2200, 2200]),
  H2("C. Noise-aware fine-tuning (RQ2)"),
  P("Fig. 2 and Table V summarise the fine-tuning study, macro-averaged over the three collections. Two findings hold across seeds and both protocols. First, fine-tuning on clean in-domain text makes the reranker more fragile to speech: the typed-to-telephone drop widens from 31.9% to 35.4% (protocol A) and 35.8% (protocol B). Second, training on transcripts recovers a substantial part of this loss: over the telephone, FT-noisy and FT-mixed outperform FT-clean on all three collections under both protocols, significantly so on NFCorpus and FiQA under protocol A and on all three under protocol B (Table VI). On typed queries, the noisy and mixed variants are not significantly different from FT-clean, and are slightly better on the macro average."),
  P("However, no fine-tuned variant outperforms the original zero-shot reranker on spoken queries. With about 1.2k training queries, fine-tuning hurts NFCorpus significantly for every variant and leaves SciFact and FiQA close to the zero-shot model. Noise-aware training with labels therefore repairs part of the damage that in-domain fine-tuning causes, but in this data regime does not add robustness beyond the original model."),
  P("Consistency distillation behaves differently. FT-distill is the only fine-tuned variant that is never significantly worse than the zero-shot model, in any collection, condition or protocol. It removes the NFCorpus damage entirely (telephone nDCG@10 0.156 vs. 0.157 zero-shot and 0.120 for FT-clean) and is significantly better than FT-clean on NFCorpus and FiQA in every condition. On the macro average it is the best reranker in all three conditions under both protocols (Table V; 0.299 vs. 0.297 over the telephone), and its seed-to-seed standard deviation is at most 0.002, against up to 0.024 for the labelled variants. Its gain over the zero-shot model is small and not significant."),
  ...fig("fig2_finetuning.png", 480, 224, "Fig. 2. Macro-averaged nDCG@10 of the zero-shot and fine-tuned rerankers, including consistency distillation (protocol B; error bars: standard deviation over three seeds)."),
  tcap("TABLE V. Macro-averaged nDCG@10 over three collections and three seeds"),
  table(["Reranker", "Protocol", "Typed", "Wideband", "Telephone", "Typed→phone"],
    [["Zero-shot (MS MARCO)", "—", "0.436", "0.379", "0.297", "−31.9%"],
     ["FT-clean (control)", "A", "0.426", "0.361", "0.275", "−35.4%"],
     ["FT-noisy", "A", "0.433", "0.371", "0.288", "−33.4%"],
     ["FT-mixed", "A", "0.431", "0.370", "0.287", "−33.4%"],
     ["FT-distill", "A", "0.440", "0.382", "0.300", "−31.9%"],
     ["FT-clean (control)", "B", "0.422", "0.355", "0.271", "−35.8%"],
     ["FT-noisy", "B", "0.425", "0.361", "0.279", "−34.3%"],
     ["FT-mixed", "B", "0.426", "0.362", "0.280", "−34.4%"],
     ["FT-distill", "B", "0.440", "0.382", "0.299", "−31.9%"]],
    [2426, 1000, 1400, 1400, 1400, 1400]),
  tcap("TABLE VI. Telephone queries, protocol B: nDCG@10 (mean ± s.d. over seeds). † p < 0.05 vs. zero-shot; ‡ p < 0.05 vs. FT-clean"),
  table(["Reranker", "SciFact", "NFCorpus", "FiQA"],
    [["Zero-shot", "0.509", "0.157", "0.226"],
     ["FT-clean", "0.487 ± 0.012", "0.120† ± 0.001", "0.205† ± 0.010"],
     ["FT-noisy", "0.504‡ ± 0.007", "0.124†‡ ± 0.003", "0.210†‡ ± 0.006"],
     ["FT-mixed", "0.503‡ ± 0.007", "0.125†‡ ± 0.004", "0.211†‡ ± 0.005"],
     ["FT-distill", "0.515 ± 0.002", "0.156‡ ± 0.000", "0.227‡ ± 0.000"]],
    [2426, 2200, 2200, 2200]),
  P("Results for the held-out voice follow the overall pattern. Over the telephone, FT-mixed (protocol B) trails the zero-shot model by 0.016 nDCG@10 on the four training voices and by 0.021 on the unseen Irish voice, so we find no evidence that the fine-tuned models simply memorise training voices.", { before: 120 }),
  H2("D. Effectiveness, latency and model size (RQ3)"),
  P("Fig. 3 relates re-ranking depth to effectiveness and latency, measured per query with batch size equal to depth on a warm model. At the production depth of 25, re-ranking takes about 1.1 s (median) on a 4-thread CPU and about 80 ms on a T4 GPU. Re-ranking 10 candidates costs about 0.4 s on CPU and retains roughly half or more of the depth-25 gain on every collection. A 12-layer cross-encoder (33.4M parameters) improves SciFact significantly (0.687 vs. 0.674, p = 0.013) but not NFCorpus or FiQA, and increasing the input length from 256 to 512 tokens gives no significant change on any collection."),
  ...fig("fig3_depth_latency.png", 600, 162, "Fig. 3. nDCG@10 (blue) and median CPU re-ranking latency (red) against re-ranking depth."),
  H2("E. Real human speech across English dialects (RQ4)"),
  P("On SD-QA, real speech reduces nDCG@10 of the production pipeline (BM25 → CE) from 0.581 on typed questions by 11.2% on average over the 21 conditions, from 3.3% for US speakers to 21.9% for South Indian speakers transcribed by the en-IN model, the worst condition in the benchmark. The four Indian-English conditions lose 16.9% on average, comparable only to Kenyan and Nigerian speech with the US model (Fig. 4). As on synthetic speech, BM25, dense, hybrid and re-ranked pipelines lose similar proportions."),
  P("The choice of ASR model matters as much as the dialect (Table VII). A localised model improves retrieval significantly for six dialects, by up to 0.037 nDCG@10 (Australia). For both Indian dialects the opposite holds: the en-IN model is significantly worse than the en-US model (−0.020 India-North, −0.026 India-South), matching its higher word error rate on these recordings."),
  ...fig("fig4_sdqa_dialects.png", 600, 233, "Fig. 4. SD-QA: nDCG@10 of BM25 → CE per dialect with the localised and the en-US ASR model. Dashed line: typed questions. Indian dialects in red."),
  tcap("TABLE VII. Localised vs. en-US ASR model, same speakers: WER and nDCG@10 of BM25 → CE (* p < 0.05, paired randomization test)"),
  table(["Dialect", "Localised model", "WER loc. / US", "nDCG loc.", "nDCG US", "Δ"],
    [["UK", "en-GB", "0.080 / 0.109", "0.543", "0.508", "+0.035*"],
     ["Australia", "en-AU", "0.082 / 0.102", "0.549", "0.513", "+0.037*"],
     ["Nigeria", "en-NG", "0.118 / 0.156", "0.505", "0.471", "+0.034*"],
     ["Kenya", "en-KE", "0.154 / 0.177", "0.488", "0.468", "+0.021*"],
     ["New Zealand", "en-NZ", "0.067 / 0.089", "0.536", "0.518", "+0.018*"],
     ["South Africa", "en-ZA", "0.069 / 0.078", "0.556", "0.546", "+0.010"],
     ["Ireland", "en-IE", "0.083 / 0.086", "0.535", "0.532", "+0.003"],
     ["Philippines", "en-PH", "0.084 / 0.076", "0.538", "0.535", "+0.003"],
     ["India-North", "en-IN", "0.129 / 0.102", "0.490", "0.510", "−0.020*"],
     ["India-South", "en-IN", "0.176 / 0.120", "0.454", "0.479", "−0.026*"]],
    [1700, 1500, 1700, 1400, 1400, 1326]),
  P("Table VIII applies the BEIR-trained rerankers to SD-QA. The labelled fine-tuned variants are within ±0.002 of the zero-shot model in every group, so the fragility caused by clean fine-tuning on BEIR does not carry over to out-of-domain Wikipedia questions. FT-distill gives the highest score in every speech group, with the largest margin on Indian speech transcribed by the en-IN model (0.475 vs. 0.472). These differences are small: the uncorrected p-value against zero-shot is 0.016–0.08, and none survives Holm correction over the seven comparisons in each group.", { before: 120 }),
  tcap("TABLE VIII. Rerankers on SD-QA real speech, BM25 top-25: nDCG@10 (fine-tuned: protocol B, mean of 3 seeds)"),
  table(["Group", "Zero-shot", "FT-clean", "FT-noisy", "FT-mixed", "FT-distill", "Raw p (distill vs. zero)"],
    [["Typed", "0.581", "0.582", "0.580", "0.581", "0.586", "0.030"],
     ["All 21 speech conditions", "0.516", "0.515", "0.516", "0.516", "0.519", "0.079"],
     ["Indian English (4)", "0.483", "0.482", "0.483", "0.484", "0.486", "0.029"],
     ["Indian English, en-IN ASR", "0.472", "0.470", "0.471", "0.471", "0.475", "0.016"],
     ["US English", "0.562", "0.564", "0.564", "0.564", "0.564", "0.279"]],
    [2226, 1000, 1000, 1000, 1000, 1100, 1700]),
  P("The first stage, not the reranker, limits the production pipeline on SD-QA (Table IX, Fig. 5). BM25 retrieves only 80.5% of relevant passages in its top 100 on real speech, against 90.2% for dense retrieval. Re-ranking a hybrid BM25 + dense list instead of the BM25 list raises nDCG@10 from 0.516 to 0.607 on real speech and from 0.472 to 0.552 for Indian speech with the en-IN model, and is significantly better than dense retrieval alone (p < 0.001). The hybrid pipeline also loses a smaller share to speech (−9.5% vs. −11.2%). Re-ranking the hybrid list with FT-distill adds a further 0.004–0.006 in every group (uncorrected p = 0.011–0.028). This comparison was not part of the original analysis plan, so we report it as exploratory.", { before: 120 }),
  tcap("TABLE IX. Choice of first stage on SD-QA: nDCG@10 (Recall@100 of the first stage in brackets)"),
  table(["Group", "BM25", "Dense", "BM25 → CE", "Hybrid → CE", "Hybrid → FT-distill"],
    [["Typed", "0.375 (0.897)", "0.615 (0.987)", "0.581", "0.671", "0.677"],
     ["All 21 speech conditions", "0.323 (0.805)", "0.537 (0.902)", "0.516", "0.607", "0.611"],
     ["Indian English (4)", "0.304 (0.771)", "0.505 (0.866)", "0.483", "0.572", "0.576"],
     ["Indian English, en-IN ASR", "0.294 (0.752)", "0.478 (0.839)", "0.472", "0.552", "0.557"]],
    [2426, 1400, 1400, 1200, 1300, 1300]),
  ...fig("fig5_sdqa_pipelines.png", 600, 233, "Fig. 5. SD-QA: production pipeline vs. dense retrieval and re-ranking of a hybrid candidate list, on typed and real spoken questions."),
  H2("F. Pre-registered follow-up: a modern reranker and which words break"),
  P("Before running further experiments we pre-registered six hypotheses, test sets and statistical tests (committed to the repository on 21 Oct 2026, before any result). All six tests were run as specified and are reported here, including the one that failed. The confirmatory test sets were new: SD-QA re-transcribed with Whisper base.en (the Part 1 recogniser; 704 questions × 11 dialects after dropping 43 questions with no Kenyan recording), HeySQuAD dev (1,002 human-spoken questions over 11,559 SQuAD paragraphs) and SLUE-SQA-5 test (2,382 real-speech questions over 3,192 documents)."),
  P([{ t: "Modern reranker (C2). ", bold: true }, "We repeated the Part 1 fine-tuning study with BAAI/bge-reranker-base (278M parameters, XLM-RoBERTa), using identical training data and hyper-parameters (fp16 mixed precision, protocol B, three seeds). Both findings replicate and the distillation result strengthens (Table X, Fig. 6). Clean fine-tuning widens the typed-to-telephone drop by 2.8 points (paired bootstrap 95% CI 0.7–4.8) and, with noisy fine-tuning, significantly hurts NFCorpus and FiQA. Consistency distillation is significantly better than the zero-shot model in 7 of the 9 dataset–condition cells and never worse, raising telephone nDCG@10 from 0.282 to 0.295 on the macro average. Its seed-to-seed standard deviation is at most 0.001. Zero-shot bge-reranker-v2-m3 (568M) loses as large a share to telephone speech as the 22.7M MiniLM (−32.3% vs. −31.9%), so model size alone does not protect against ASR errors; re-ranking a hybrid list with it loses the least (−27.3%)."]),
  tcap("TABLE X. bge-reranker-base on BEIR (macro over three collections; protocol B, mean of 3 seeds)"),
  table(["Reranker", "Typed", "Wideband", "Telephone", "Typed→phone", "Δ drop vs. zero-shot [95% CI]"],
    [["Zero-shot", "0.420", "0.365", "0.282", "−32.9%", "—"],
     ["FT-clean", "0.421", "0.362", "0.271", "−35.7%", "−2.8 [−4.8, −0.7]"],
     ["FT-noisy", "0.413", "0.357", "0.271", "−34.3%", "−1.4 [−3.5, +0.6]"],
     ["FT-distill", "0.431", "0.376", "0.295", "−31.5%", "+1.4 [0.0, +2.7]"]],
    [1800, 1100, 1200, 1200, 1300, 2426]),
  ...fig("fig6_bge_finetuning.png", 480, 224, "Fig. 6. bge-reranker-base: zero-shot vs. fine-tuned variants (macro nDCG@10; error bars: s.d. over three seeds)."),
  P([{ t: "Which words break (C4). ", bold: true }, "We aligned each question to its transcript and tagged its words with spaCy. Regressing the per-query change in nDCG@10 on the error rates of content words (nouns, proper nouns, numbers, adjectives, main verbs) and function words shows that content-word errors drive the loss (Table XI, Fig. 7): on SD-QA, a fully misrecognised content vocabulary costs 0.65 nDCG@10, while function-word errors have no harmful effect. A misrecognised named entity alone costs 0.27 nDCG@10 on SD-QA. The difference between the two coefficients is significant in every group (query-cluster bootstrap)."]),
  P([{ t: "Where the loss happens (C5). ", bold: true }, "Among spoken queries that lose effectiveness, the relevant passage is absent from the BM25 top-25 in 68% (SD-QA with Whisper, 95% CI 64–72%), 53% (SD-QA with Google ASR) and 56% (BEIR telephone, 51–61%) of cases, and 42% for wideband BEIR; the pre-registered prediction (> 50% on SD-QA-Whisper and BEIR telephone) holds. In these cases the reranker never sees the right passage, which explains why first-stage improvements help more than re-ranking."]),
  tcap("TABLE XI. Effect of word errors on nDCG@10 (OLS coefficients, 95% query-cluster bootstrap CI) and share of losses caused by first-stage recall failure"),
  table(["Group", "β content words", "β function words", "Recall-failure share"],
    [["SD-QA (Whisper base.en)", "−0.639 [−0.716, −0.569]", "+0.046 [−0.016, +0.111]", "68% [64, 72]"],
     ["SD-QA (Google ASR)", "−0.646 [−0.745, −0.553]", "+0.117 [+0.034, +0.213]", "53% [47, 59]"],
     ["BEIR wideband", "−0.369 [−0.498, −0.252]", "−0.025 [−0.134, +0.069]", "42% [35, 50]"],
     ["BEIR telephone", "−0.363 [−0.443, −0.285]", "−0.011 [−0.086, +0.061]", "56% [51, 61]"]],
    [2200, 2400, 2400, 2026]),
  ...fig("fig7_word_errors.png", 520, 236, "Fig. 7. Change in nDCG@10 per unit error rate of content vs. function words (95% CI)."),
  P([{ t: "First stage on new real speech (C1, confirmed). ", bold: true }, "Re-ranking a hybrid candidate list beats re-ranking a BM25 list on all three new test sets (Table XII, Fig. 8), each significant after Holm correction: +0.085 nDCG@10 on SD-QA-Whisper, +0.046 on SLUE-SQA-5 and +0.010 on HeySQuAD, where BM25 alone already scores 0.85. The Part 2 exploratory finding therefore replicates on unseen data and a different recogniser."]),
  tcap("TABLE XII. C1: BM25 → ce-l6 vs. Hybrid → ce-l6 on new real-speech test sets (nDCG@10; Holm over 3 tests)"),
  table(["Test set", "Queries", "BM25", "Dense", "BM25 → CE", "Hybrid → CE", "Δ", "p (Holm)"],
    [["SD-QA-Whisper", "704", "0.286", "0.489", "0.469", "0.554", "+0.085", "<0.001"],
     ["HeySQuAD dev", "1,002", "0.853", "0.740", "0.896", "0.906", "+0.010", "0.023"],
     ["SLUE-SQA-5 test", "2,382", "0.332", "0.393", "0.469", "0.515", "+0.046", "<0.001"]],
    [1700, 900, 900, 900, 1100, 1200, 1000, 1326]),
  ...fig("fig9_first_stage_new_sets.png", 560, 206, "Fig. 8. Spoken-query nDCG@10 on the new real-speech test sets for BM25 and hybrid candidate lists re-ranked by MiniLM (ce-l6) and bge-reranker-v2-m3."),
  P([{ t: "Real vs. synthetic training noise (C3, not supported). ", bold: true }, "We predicted that training on real human speech errors would transfer to real accented speech better than training on synthetic TTS errors. The opposite held (Table XIII, Fig. 9). On the same 1,985 HeySQuAD questions, models trained on synthetic-speech transcripts were never worse than those trained on real-speech transcripts, and on SLUE-SQA-5 they were significantly better (FT-noisy: 0.447 vs. 0.385; distillation: 0.467 vs. 0.421; p < 0.001). The real transcripts had twice the word error rate of the synthetic ones (0.150 vs. 0.075), so more of them no longer expressed the question; label noise, not realism, appears to dominate at this scale. Labelled fine-tuning on HeySQuAD again fell below the zero-shot model on every out-of-domain set, while distillation was the only variant to exceed it (on SD-QA-Whisper)."]),
  tcap("TABLE XIII. C3: ce-l6 trained on 1,985 HeySQuAD questions (nDCG@10, mean of 3 seeds). Real vs. synthetic: Holm over 6 tests"),
  table(["Test set", "Zero-shot", "FT-clean", "FT-noisy real", "FT-noisy synth", "Distill real", "Distill synth"],
    [["SD-QA-Whisper", "0.469", "0.452", "0.455", "0.459", "0.473", "0.475"],
     ["SD-QA-Whisper, Indian", "0.448", "0.434", "0.436", "0.441", "0.451", "0.453"],
     ["SLUE-SQA-5 test", "0.469", "0.450", "0.385‡", "0.447", "0.421‡", "0.467"],
     ["HeySQuAD dev (in-domain)", "0.896", "0.906", "0.904", "0.908", "0.902", "0.904"]],
    [2026, 1100, 1100, 1200, 1200, 1200, 1200]),
  P("‡ significantly worse than the synthetic counterpart (p < 0.001 after Holm). No comparison favoured real-speech training.", { align: AlignmentType.CENTER }),
  ...fig("fig8_real_vs_synth.png", 600, 226, "Fig. 9. Training on real vs. synthetic speech errors (HeySQuAD-trained MiniLM)."),
  P([{ t: "Stronger distillation (C6, confirmed). ", bold: true }, "On validation only, we compared a 568M-parameter teacher (bge-reranker-v2-m3) with BM25 or hybrid candidates and temperatures 1, 2 and 4, and the best of these trained on BEIR plus HeySQuAD. The selected configuration (bge-reranker-v2-m3 teacher, hybrid candidates, τ = 4, BEIR training data) was retrained with three seeds and tested once. The 22.7M MiniLM student now significantly beats its zero-shot self on all three pre-registered test groups (Table XIV): this is the first significant improvement over zero-shot for the small model in the whole study."]),
  tcap("TABLE XIV. C6: selected distillation configuration vs. zero-shot ce-l6 (nDCG@10, mean of 3 seeds; Holm over 3 tests)"),
  table(["Test set", "Zero-shot ce-l6", "Distilled ce-l6", "Δ", "p (Holm)"],
    [["BEIR spoken (3 datasets × 2 channels)", "0.313", "0.321", "+0.008", "<0.001"],
     ["SD-QA-Whisper", "0.469", "0.476", "+0.007", "0.004"],
     ["SLUE-SQA-5 test", "0.469", "0.475", "+0.006", "0.004"]],
    [3226, 1600, 1600, 1100, 1500]),
  P([{ t: "Stronger ASR (descriptive). ", bold: true }, "Re-transcribing SD-QA with Whisper large-v3-turbo instead of base.en cuts the word error rate by more than half in every dialect (e.g. India-South 0.190 → 0.075) and recovers most of the loss: BM25 → CE on India-South rises from 0.438 to 0.531, against 0.582 for typed questions. The Indian-English gap narrows from 0.10 to about 0.03 nDCG@10 relative to US speakers."]),
);

// ---------------------------------------------------------------- V–VIII
add(H1("V. Discussion"),
  P([{ t: "Why the reranker does not help on speech. ", bold: true }, "A cross-encoder reads the query and passage together, but it cannot restore information that the transcript has lost. When a key term is replaced by another real word, both the first stage and the reranker match the wrong term, which explains why all pipelines lose a similar proportion of effectiveness."]),
  P([{ t: "Why clean fine-tuning increases fragility. ", bold: true }, "Fine-tuning on clean in-domain queries appears to strengthen reliance on exact domain terminology, which is precisely what ASR corrupts. Exposure to transcripts during training counteracts this, which is consistent with typo-aware training for text noise [5]."]),
  P([{ t: "Why fine-tuning hurts NFCorpus. ", bold: true }, "NFCorpus queries have many relevant documents each, and its relevance judgments are incomplete. Hard negatives mined from BM25 are therefore likely to include unjudged relevant documents, teaching the model to reject relevant passages. Consistency distillation provides indirect support for this explanation, now replicated on bge-reranker-base: it uses the same queries and candidate passages but no relevance labels, and it removes the NFCorpus damage completely."]),
  P([{ t: "Why distillation is safe but small. ", bold: true }, "Distillation anchors the student to the original model's ranking of each candidate list, so it cannot drift far from the zero-shot model; this explains both its stability across seeds and its small effect. It teaches the model to give a transcript the same scores as the question it came from, which helps when the transcript still contains the key terms, but cannot help when ASR has removed them."]),
  P([{ t: "Real speech and Indian English. ", bold: true }, "Real speech is milder than our simulated telephone channel but still costs every pipeline about a tenth of its effectiveness, and Indian-English speakers lose the most. The en-IN recogniser in SD-QA's 2021 transcripts produces more word errors on Indian speech than the en-US model, so a deployment should choose its ASR model by measured accuracy on its own callers rather than by locale name."]),
  P([{ t: "Where the loss happens. ", bold: true }, "On SD-QA's naturally phrased questions BM25 misses many relevant passages, and ASR errors widen the gap. A reranker can only reorder what the first stage returns, so improving the candidate list helps far more than any change to the reranker, consistent with SQuTR's finding that the retrieval backend matters most [2]."]),
  P([{ t: "Implications for voice agents. ", bold: true }, "(1) Speech quality, not the reranker, is the main lever: telephone audio costs 24–58% of effectiveness. (2) Re-rank a hybrid BM25 + dense candidate list rather than a BM25 list. (3) If a reranker is adapted for a deployment, prefer label-free consistency distillation, or at least include transcripts in training; clean fine-tuning makes it more fragile. (4) Choose the ASR model by measured accuracy for the callers it will serve, especially for Indian English. (5) On CPU, the re-ranking depth must be reduced to fit a live turn, or the reranker moved to a GPU."]),
);

add(H1("VI. Limitations"),
  bullet("The training speech is synthetic. Real speech is evaluated only on SD-QA, whose transcripts come from Google's 2021 recogniser rather than Whisper, and whose Wikipedia questions differ in domain from the BEIR collections."),
  bullet("Two reranker families (MiniLM, BGE) are fine-tuned; the SD-QA, HeySQuAD and SLUE recordings are not telephone audio."),
  bullet("Distillation gains on MiniLM are small in absolute terms (+0.006 to +0.008 nDCG@10 with the stronger teacher), although significant on all pre-registered test sets."),
  bullet("The real-speech training set (1,985 HeySQuAD questions) is small, and its real transcripts are noisier than its synthetic ones, so C3 compares noise levels as well as noise types."),
  bullet("The training set is small (1,184 queries), and hard negatives may contain unjudged relevant documents, particularly on NFCorpus."),
  bullet("Latency is measured on Kaggle hardware (4-thread x86 CPU, Tesla T4), not on the deployment machine."),
  bullet("The telephone channel is simulated; real calls add codec artefacts and packet loss."),
);

add(H1("VII. Future Work"),
  P("We plan to (1) re-rank several ASR hypotheses (an N-best list) instead of one transcript, attacking both content-word errors and first-stage recall failures; (2) train on larger and cleaner real-speech data, filtering transcripts by word error rate, to separate noise type from noise level; (3) distil into larger students; (4) record real Indian-English telephone queries; and (5) deploy the hybrid first stage with a distilled reranker in the voice agent."),
);

add(H1("VIII. Conclusion"),
  P("Spoken queries reduce retrieval effectiveness by 7–25% over wideband audio, 24–58% over a telephone channel, and 11–19% on real speech, with Indian English among the hardest hit; a cross-encoder reranker improves absolute effectiveness but does not reduce this relative loss. Misrecognised content words drive the loss, and in more than half of the failures the relevant passage never reaches the reranker. Fine-tuning the reranker on clean text makes it more fragile to speech, on two model families. Label-free consistency distillation avoids this damage and, with a stronger teacher, significantly improves on the zero-shot reranker on every pre-registered test set; training on real rather than synthetic speech errors did not help at our scale. The largest gains come from outside the reranker: a hybrid first stage (confirmed on three new real-speech test sets) and a stronger recogniser. Robustness work for voice agents should therefore target the recogniser, the first stage and the training objective together."),
);

add(H1("Acknowledgement and Use of AI Tools"),
  P("[Describe here the use of AI tools in this project, as required by the course GenAI policy. Experiments were run on Kaggle (NVIDIA Tesla T4). Code, data processing and evaluation scripts are in the project repository.]"),
);

// ---------------------------------------------------------------- references
const refs = [
  "G. Sidiropoulos, S. Vakulenko, and E. Kanoulas, \"On the impact of speech recognition errors in passage retrieval for spoken question answering,\" in Proc. CIKM, 2022, doi: 10.1145/3511808.3557662.",
  "Y. Li et al., \"SQuTR: A robustness benchmark for spoken query to text retrieval under acoustic noise,\" arXiv:2602.12783, 2026.",
  "Z. Bao, \"Better retrieval, worse robustness: How multi-hop RAG amplifies upstream ASR errors,\" in Proc. EMNLP, 2026, arXiv:2608.22872.",
  "Y. Moriya and G. J. F. Jones, \"Improving noise robustness for spoken content retrieval using semi-supervised ASR and N-best transcripts for BERT-based ranking models,\" in Proc. IEEE SLT, 2022, arXiv:2301.06056.",
  "S. Zhuang and G. Zuccon, \"Dealing with typos for BERT-based passage retrieval and ranking,\" in Proc. EMNLP, 2021, pp. 2836–2842.",
  "S. Zhuang and G. Zuccon, \"CharacterBERT and self-teaching for improving the robustness of dense retrievers on queries with typos,\" in Proc. SIGIR, 2022.",
  "X. Ma, C. dos Santos, and A. O. Arnold, \"Contrastive fine-tuning improves robustness for neural rankers,\" in Findings of ACL-IJCNLP, 2021, pp. 570–582.",
  "G. Penha, A. Câmara, and C. Hauff, \"Evaluating the robustness of retrieval pipelines with query variation generators,\" in Proc. ECIR, LNCS 13185, 2022, pp. 397–412.",
  "T. Hagen, H. Scells, and M. Potthast, \"Revisiting query variation robustness of transformer models,\" in Findings of EMNLP, 2024.",
  "G. M. Rosa et al., \"In defense of cross-encoders for zero-shot retrieval,\" arXiv:2212.06121, 2022.",
  "N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, \"BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models,\" in NeurIPS Datasets and Benchmarks Track, 2021.",
  "R. Nogueira and K. Cho, \"Passage re-ranking with BERT,\" arXiv:1901.04085, 2019.",
  "A. Radford et al., \"Robust speech recognition via large-scale weak supervision,\" in Proc. ICML, 2023.",
  "A. Koenecke et al., \"Careless Whisper: Speech-to-text hallucination harms,\" in Proc. ACM FAccT, 2024, arXiv:2402.08021.",
  "C.-H. Lee et al., \"Spoken SQuAD: A study of mitigating the impact of speech recognition errors on listening comprehension,\" in Proc. Interspeech, 2018.",
  "F. Faisal, S. Keshava, M. M. I. Alam, and A. Anastasopoulos, \"SD-QA: Spoken dialectal question answering for the real world,\" in Findings of EMNLP, 2021.",
  "G. V. Cormack, C. L. A. Clarke, and S. Büttcher, \"Reciprocal rank fusion outperforms Condorcet and individual rank learning methods,\" in Proc. SIGIR, 2009.",
  "S. Xiao et al., \"C-Pack: Packaged resources to advance general Chinese embedding,\" arXiv:2309.07597, 2023.",
  "W. Wang et al., \"MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,\" in Proc. NeurIPS, 2020.",
  "M. D. Smucker, J. Allan, and B. Carterette, \"A comparison of statistical significance tests for information retrieval evaluation,\" in Proc. CIKM, 2007.",
  "I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in Proc. ICLR, 2019.",
  "Y. Wu et al., \"HeySQuAD: A spoken question answering dataset,\" arXiv:2304.13689, 2023.",
  "T. Javed et al., \"Svarah: Evaluating English ASR systems on Indian accents,\" in Proc. Interspeech, 2023.",
  "S. Holm, \"A simple sequentially rejective multiple test procedure,\" Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.",
];
add(H1("References"), refs.map((r, i) => new Paragraph({
  children: [t(`[${i + 1}] `, { size: 19 }), t(r, { size: 19 })], spacing: { after: 60 },
  indent: { left: 440, hanging: 440 }, alignment: AlignmentType.LEFT,
})));

const doc = new Document({
  creator: "Arya Shah, Teerth Shah, Urja Singhi", title: "Does the Reranker Hear You?",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });
