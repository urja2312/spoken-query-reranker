"""Build the noise-aware fine-tuning set from TRAIN-split queries only.

For each sampled train query we keep two surface forms of the same information need:
  * `clean` — the original text;
  * `noisy` — a TTS -> ASR transcript, through the wideband or the telephone channel (chosen
    per query, 50/50), produced by exactly the pipeline used for the test conditions.
Hard negatives are mined with BM25 *using the form that will be trained on*, because at test
time the re-ranker sees a pool retrieved by the noisy query, not the clean one.

Leakage controls: test query ids are disjoint from train (checked), and any train query whose
normalised text equals a test query's text is dropped. One TTS voice (Moira, en_IE) is held out
of training entirely, so test results can be split into seen vs unseen voices.
Usage: python -I build_trainset.py
"""

from __future__ import annotations

import random
import re
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech  # noqa: E402
from bench.first_stage import BM25  # noqa: E402

PER_DATASET = {"train": 400, "val": 40}
NEG_POOL = 30
TRAIN_VOICES = tuple(v for v in speech.VOICES if v != "Moira")     # Moira is the unseen voice
OUT = data.CACHE / "trainset.json"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def main() -> None:
    built = data.load_json(OUT, {"examples": []})
    done = {(e["dataset"], e["qid"]) for e in built["examples"]}
    recognizer = speech.Recognizer("base.en")
    for name in data.DATASETS:
        ds = data.load(name)
        test_text = {norm(t) for t in ds.queries.values()}
        queries, qrels = data.load_split(name, "train")
        pool = sorted(q for q in queries if norm(queries[q]) not in test_text)
        dropped = len(queries) - len(pool)
        rng = random.Random(f"trainset:{name}")
        picked = rng.sample(pool, PER_DATASET["train"] + PER_DATASET["val"])
        roles = {q: ("val" if i < PER_DATASET["val"] else "train") for i, q in enumerate(picked)}
        bm25 = BM25(ds)
        audio_dir = HERE / "audio" / f"{name}-train"
        print(f"[{name}] {len(queries)} train queries, {dropped} dropped as text-duplicates of test; "
              f"sampling {len(picked)}", flush=True)
        started, n_new = time.time(), 0
        for qid in picked:
            if (name, qid) in done:
                continue
            seed = zlib.crc32(f"train:{name}:{qid}".encode())
            voice = TRAIN_VOICES[seed % len(TRAIN_VOICES)]
            wav = audio_dir / f"{qid}.wav"
            if not wav.exists():
                speech.synthesize(queries[qid], voice, wav)
            clip = speech.read_wav(wav)
            channel = "asr_phone" if seed % 2 else "asr_clean"
            if channel == "asr_phone":
                others = [speech.read_wav(audio_dir / f"{o}.wav") for o in picked[:5]
                          if o != qid and (audio_dir / f"{o}.wav").exists()]
                clip = speech.telephone(clip, others or [clip[::-1]], seed)
            noisy = recognizer(clip)
            forms = {"clean": queries[qid], "noisy": noisy}
            negatives = {form: [d for d, _ in bm25.search(text, NEG_POOL) if d not in qrels[qid]]
                         for form, text in forms.items()}
            built["examples"].append({
                "dataset": name, "qid": qid, "role": roles[qid], "voice": voice, "channel": channel,
                "clean": forms["clean"], "noisy": noisy, "wer": speech.wer(forms["clean"], noisy),
                "positives": qrels[qid], "negatives": negatives})
            n_new += 1
            if n_new % 25 == 0:
                data.save_json(OUT, built)
                print(f"[{name}] {n_new} built ({n_new / (time.time() - started):.2f} q/s)", flush=True)
        data.save_json(OUT, built)
    ex = built["examples"]
    print(f"done: {len(ex)} examples "
          f"({sum(e['role'] == 'train' for e in ex)} train / {sum(e['role'] == 'val' for e in ex)} val), "
          f"mean training WER {sum(e['wer'] for e in ex) / len(ex):.3f}", flush=True)


if __name__ == "__main__":
    main()
