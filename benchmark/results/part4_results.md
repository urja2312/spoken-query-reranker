# Part 4 results: Listen Twice and Mishear the Corpus

Primary metric: nDCG@10 of Hybrid > ce-l6 @25 (pre-registered in paper/PREREGISTRATION_PART4.md). SD-QA scores are per-question means over 11 dialects (43 Kenyan questions without audio dropped).

## Pre-registered tests (paired randomization, Holm over all tests)

| Test | Dataset | A | B | Δ (A − B) [95% CI] | p | p (Holm) | Supported |
|---|---|---|---|---|---|---|---|
| H1: Listen Twice > pass 1 | SD-QA (11 dialects) | 0.5711 | 0.5538 | +0.0172 [+0.0116, +0.0232] | 0.0001 | 0.0012 | yes |
| H1: Listen Twice > pass 1 | HeySQuAD | 0.9119 | 0.9059 | +0.0060 [-0.0004, +0.0127] | 0.0774 | 0.6191 | no |
| H1: Listen Twice > pass 1 | SLUE-SQA-5 | 0.5177 | 0.5146 | +0.0031 [-0.0016, +0.0079] | 0.2021 | 1.0000 | no |
| H2: Listen Twice > random-passage prompt | SD-QA (11 dialects) | 0.5711 | 0.5543 | +0.0168 [+0.0108, +0.0230] | 0.0001 | 0.0012 | yes |
| H2: Listen Twice > random-passage prompt | HeySQuAD | 0.9119 | 0.8988 | +0.0131 [+0.0061, +0.0207] | 0.0003 | 0.0030 | yes |
| H2: Listen Twice > random-passage prompt | SLUE-SQA-5 | 0.5177 | 0.5093 | +0.0083 [+0.0033, +0.0136] | 0.0009 | 0.0081 | yes |
| H3: Mishear the Corpus > plain index | SD-QA (11 dialects) | 0.5546 | 0.5538 | +0.0007 [-0.0021, +0.0035] | 0.6217 | 1.0000 | no |
| H3: Mishear the Corpus > plain index | HeySQuAD | 0.9071 | 0.9059 | +0.0011 [-0.0005, +0.0037] | 0.3732 | 1.0000 | no |
| H3: Mishear the Corpus > plain index | SLUE-SQA-5 | 0.5156 | 0.5146 | +0.0010 [-0.0015, +0.0036] | 0.4347 | 1.0000 | no |
| H4: Mishear the Corpus > phonetic index | SD-QA (11 dialects) | 0.5546 | 0.5575 | -0.0030 [-0.0076, +0.0016] | 0.2193 | 1.0000 | no |
| H4: Mishear the Corpus > phonetic index | HeySQuAD | 0.9071 | 0.9063 | +0.0008 [-0.0026, +0.0042] | 0.6717 | 1.0000 | no |
| H4: Mishear the Corpus > phonetic index | SLUE-SQA-5 | 0.5156 | 0.5152 | +0.0004 [-0.0029, +0.0037] | 0.8159 | 1.0000 | no |

## All systems (mean nDCG@10; Recall@25 of the hybrid pool)

| System | SD-QA (11 dialects) ce-l6 | SD-QA (11 dialects) bge-m3 | SD-QA (11 dialects) R@25 | HeySQuAD ce-l6 | HeySQuAD bge-m3 | HeySQuAD R@25 | SLUE-SQA-5 ce-l6 | SLUE-SQA-5 bge-m3 | SLUE-SQA-5 R@25 |
|---|---|---|---|---|---|---|---|---|---|
| Typed question (reference) | 0.6694 | 0.7476 | 0.9001 | 0.9378 | 0.9510 | 0.9870 | 0.5630 | 0.5317 | 0.8904 |
| Typed, MC index | 0.6707 | — | 0.9015 | 0.9375 | — | 0.9870 | 0.5629 | — | 0.8858 |
| Typed, phonetic index | 0.6697 | — | 0.9058 | 0.9395 | — | 0.9890 | 0.5634 | — | 0.8841 |
| Spoken, pass 1 (baseline) | 0.5538 | 0.6125 | 0.7666 | 0.9059 | 0.9145 | 0.9631 | 0.5146 | 0.5006 | 0.8262 |
| Phonetic index (classic baseline) | 0.5575 | — | 0.7762 | 0.9063 | — | 0.9621 | 0.5152 | — | 0.8216 |
| **Mishear the Corpus (MC)** | 0.5546 | 0.6131 | 0.7670 | 0.9071 | 0.9169 | 0.9661 | 0.5156 | 0.5008 | 0.8237 |
| Pass 2, random-passage prompt (control) | 0.5543 | — | 0.7675 | 0.8988 | — | 0.9611 | 0.5093 | — | 0.8270 |
| **Listen Twice (LT)** | 0.5711 | 0.6352 | 0.7868 | 0.9119 | 0.9231 | 0.9691 | 0.5177 | 0.4972 | 0.8270 |
| LT-fuse (pass 1 + pass 2 runs) | 0.5727 | — | 0.7916 | 0.9094 | — | 0.9691 | 0.5200 | — | 0.8321 |
| **LT + MC** | 0.5760 | 0.6373 | 0.7909 | 0.9136 | 0.9249 | 0.9711 | 0.5181 | 0.4973 | 0.8279 |
| Pass 2, relevant-passage prompt (oracle bound) | 0.6295 | — | 0.8549 | 0.9303 | — | 0.9830 | 0.5379 | — | 0.8501 |

## Transcripts: WER, rare-word error, prompt injection

| Dataset | Transcript | WER | Rare-word error | Queries with injected prompt words | Injected words / query |
|---|---|---|---|---|---|
| SD-QA (11 dialects) | base | 0.156 | 0.378 | — | — |
| SD-QA (11 dialects) | lt | 0.141 | 0.262 | 2.0% | 0.02 |
| SD-QA (11 dialects) | rnd | 0.163 | 0.379 | 0.5% | 0.01 |
| SD-QA (11 dialects) | orc | 0.109 | 0.154 | 2.3% | 0.02 |
| SD-QA (11 dialects) | ltmc | 0.138 | 0.254 | 2.5% | 0.03 |
| HeySQuAD | base | 0.079 | 0.203 | — | — |
| HeySQuAD | lt | 0.071 | 0.151 | 0.8% | 0.01 |
| HeySQuAD | rnd | 0.085 | 0.208 | 0.1% | 0.00 |
| HeySQuAD | orc | 0.059 | 0.110 | 1.9% | 0.02 |
| HeySQuAD | ltmc | 0.072 | 0.154 | 1.0% | 0.01 |
| SLUE-SQA-5 | base | 0.103 | 0.274 | — | — |
| SLUE-SQA-5 | lt | 0.105 | 0.249 | 1.1% | 0.01 |
| SLUE-SQA-5 | rnd | 0.111 | 0.272 | 0.4% | 0.00 |
| SLUE-SQA-5 | orc | 0.098 | 0.231 | 1.2% | 0.01 |
| SLUE-SQA-5 | ltmc | 0.105 | 0.250 | 1.3% | 0.01 |

## Confirmation-bias check: LT − pass 1, split by whether pass-1 hybrid top-5 held a relevant passage

| Dataset | Pass-1 top-5 had it: n, Δ | Pass-1 top-5 missed it: n, Δ | LT helped / hurt (share of questions) | MC helped / hurt |
|---|---|---|---|---|
| SD-QA (11 dialects) | 525, +0.0190 | 179, +0.0121 | 42.3% / 21.2% | 10.1% / 7.5% |
| HeySQuAD | 902, -0.0007 | 100, +0.0664 | 2.4% / 1.5% | 0.3% / 0.1% |
| SLUE-SQA-5 | 1302, -0.0044 | 1080, +0.0121 | 5.0% / 5.4% | 2.6% / 1.8% |

## Typed queries on the MC index (non-inferiority, margin −0.005)

| Dataset | Typed, plain | Typed, MC | Δ [95% CI] | Non-inferior |
|---|---|---|---|---|
| SD-QA (11 dialects) | 0.6694 | 0.6707 | +0.0013 [-0.0034, +0.0062] | yes |
| HeySQuAD | 0.9378 | 0.9375 | -0.0003 [-0.0009, +0.0000] | yes |
| SLUE-SQA-5 | 0.5630 | 0.5629 | -0.0001 [-0.0024, +0.0023] | yes |

## LT + MC against each method alone (secondary)

| Dataset | LT+MC − LT [95% CI], p | LT+MC − MC [95% CI], p |
|---|---|---|
| SD-QA (11 dialects) | +0.0049 [+0.0015, +0.0086], p=0.0058 | +0.0214 [+0.0156, +0.0275], p=0.0001 |
| HeySQuAD | +0.0017 [-0.0011, +0.0050], p=0.3066 | +0.0065 [-0.0001, +0.0136], p=0.0652 |
| SLUE-SQA-5 | +0.0005 [-0.0035, +0.0043], p=0.8147 | +0.0026 [-0.0021, +0.0073], p=0.2871 |

## SD-QA per dialect (nDCG@10, Hybrid > ce-l6 @25)

| Dialect | Pass 1 | LT | MC | LT+MC | Oracle prompt |
|---|---|---|---|---|---|
| aus | 0.5719 | 0.5892 | 0.5731 | 0.6016 | 0.6374 |
| gbr | 0.5805 | 0.5864 | 0.5811 | 0.5924 | 0.6412 |
| ind_n | 0.5480 | 0.5721 | 0.5482 | 0.5810 | 0.6244 |
| ind_s | 0.5135 | 0.5390 | 0.5092 | 0.5369 | 0.6004 |
| irl | 0.5959 | 0.6124 | 0.6019 | 0.6145 | 0.6547 |
| kenya | 0.4974 | 0.5147 | 0.4956 | 0.5189 | 0.5942 |
| nga | 0.4298 | 0.4600 | 0.4283 | 0.4611 | 0.5834 |
| nzl | 0.5909 | 0.6061 | 0.5945 | 0.6124 | 0.6542 |
| phl | 0.5537 | 0.5731 | 0.5552 | 0.5750 | 0.6351 |
| usa | 0.6246 | 0.6352 | 0.6278 | 0.6398 | 0.6591 |
| zaf | 0.5857 | 0.5935 | 0.5853 | 0.6020 | 0.6400 |

## HeySQuAD ablation: number of passages K behind the LT prompt (post hoc)

| K | nDCG@10 |
|---|---|
| 1 | 0.9105 |
| 3 | 0.9074 |
| 5 (pre-registered) | 0.9119 |
| 10 | 0.9082 |

## Cost

| Dataset | MC: words misheard / vocabulary | MC: index growth | LT: added Whisper time per query (s) | Pass-1 re-decode identical |
|---|---|---|---|---|
| SD-QA (11 dialects) | 26,123 / 40,000 (65%) | +11.3% tokens | 0.046 | 100% |
| HeySQuAD | 27,182 / 40,000 (68%) | +12.0% tokens | 0.053 | 100% |
| SLUE-SQA-5 | 10,697 / 17,504 (61%) | +13.0% tokens | 0.054 | 100% |

Misheard-word examples (most frequent corpus words first):

- SD-QA (11 dialects): powell → howl; mycenaean → cinean / mycinean; vietnamese → attendees / theattendees; pottery → hottery; charlemagne → charles / amagnie / charlesamagnie / charlie / maggie / charliemaggie; enoch → enough; venice → thenis; conrad → comrade
- HeySQuAD: orthodox → worth / docks / worththedocks / dogs / orfordogs; madonna → medhala; schwarzenegger → war / nager / whichwarisanager / wars / egger / warsandegger; dialect → dialicked; alps → bobs; treaties → treats; plymouth → play / math / playmath; chopin → chop
- SLUE-SQA-5: luther → lother; stalin → tullen / starling; quebec → quiver; monarch → monet; spacecraft → basecraft; anglo → angle; zinc → saint / zing; polynomial → holy / nomel / holynomel
