"""Build spoken-query transcripts for every test query: asr_clean and asr_phone.

Resumable: audio and transcripts are cached, and progress is saved every 25 queries.
Usage: python -I run_speech.py [dataset ...]
"""

from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import data, speech  # noqa: E402


def main(names: list[str]) -> None:
    recognizer = speech.Recognizer("base.en")
    for name in names:
        ds = data.load(name)
        cache = data.cache_dir(name)
        audio_dir = HERE / "audio" / name
        out = {cond: data.load_json(cache / f"transcripts_{cond}.json", {})
               for cond in ("asr_clean", "asr_phone")}
        qids = sorted(ds.queries)
        started = time.time()

        # 1. Synthesise every query first: the phone condition needs other talkers for babble.
        for qid in qids:
            wav = audio_dir / f"{qid}.wav"
            if not wav.exists():
                speech.synthesize(ds.queries[qid], speech.voice_for(qid), wav)
        clips = {qid: speech.read_wav(audio_dir / f"{qid}.wav") for qid in qids}

        # 2. Transcribe both conditions.
        for n, qid in enumerate(qids, start=1):
            seed = zlib.crc32(f"{name}:{qid}".encode())
            if qid not in out["asr_clean"]:
                out["asr_clean"][qid] = recognizer(clips[qid])
            if qid not in out["asr_phone"]:
                rng_pick = [qids[(seed + 7919 * k) % len(qids)] for k in range(1, 6)]
                others = [clips[o] for o in rng_pick if o != qid]
                out["asr_phone"][qid] = recognizer(speech.telephone(clips[qid], others, seed))
            if n % 25 == 0 or n == len(qids):
                for cond, transcripts in out.items():
                    data.save_json(cache / f"transcripts_{cond}.json", transcripts)
                rate = n / max(1e-9, time.time() - started)
                print(f"[{name}] {n}/{len(qids)} queries  ({rate:.1f} q/s)", flush=True)

        for cond, transcripts in out.items():
            wers = [speech.wer(ds.queries[q], transcripts[q]) for q in qids]
            print(f"[{name}] {cond}: mean WER = {sum(wers) / len(wers):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(data.DATASETS))
