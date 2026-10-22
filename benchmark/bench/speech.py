"""Spoken queries: text -> TTS -> (optional telephone channel) -> ASR transcript.

Two acoustic conditions model the two places a voice agent's query comes from:

* `asr_clean` — wideband 16 kHz audio, as from an app or browser microphone.
* `asr_phone` — the same audio through a simulated PSTN call: 8 kHz, 300–3400 Hz band-pass,
  G.711 mu-law companding, and background babble at 10 dB SNR built from *other* queries'
  speech (several talkers in the room). That is the channel a phone-line agent actually hears.

Speech is synthesised with macOS voices rotated across five English accents, including Indian
English, so recognition errors are not one voice's quirks. Synthetic speech is cleaner than real
callers; that is a stated limitation, not a hidden one.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

VOICES = ("Rishi", "Samantha", "Daniel", "Karen", "Moira")   # en_IN, en_US, en_GB, en_AU, en_IE
RATE = 16000
PHONE_SNR_DB = 10.0
BABBLE_TALKERS = 3


def voice_for(qid: str) -> str:
    return VOICES[zlib.crc32(qid.encode()) % len(VOICES)]


def synthesize(text: str, voice: str, path: Path) -> None:
    """Text goes in on stdin, never argv, so a query starting with '-' cannot become a flag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["say", "-v", voice, "-o", str(path), "--file-format=WAVE",
                    f"--data-format=LEI16@{RATE}"], input=text.encode("utf-8"),
                   check=True, timeout=120)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == RATE and handle.getnchannels() == 1
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def _mulaw(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 mu-law encode -> 8-bit quantise -> decode."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
    return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu


def telephone(speech: np.ndarray, others: list[np.ndarray], seed: int) -> np.ndarray:
    """Simulated PSTN channel with babble noise. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    babble = np.zeros_like(speech)
    for other in others[:BABBLE_TALKERS]:
        if other.size == 0:
            continue
        start = int(rng.integers(0, max(1, other.size)))
        clip = np.roll(other, -start)
        reps = int(np.ceil(speech.size / clip.size))
        babble += np.tile(clip, reps)[: speech.size]
    p_speech = float(np.mean(speech ** 2)) or 1e-12
    p_noise = float(np.mean(babble ** 2))
    if p_noise > 0:
        babble *= np.sqrt(p_speech / (p_noise * 10 ** (PHONE_SNR_DB / 10.0)))
    narrow = resample_poly(speech + babble, 1, 2)                       # 16 kHz -> 8 kHz
    narrow = sosfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), narrow)
    peak = float(np.max(np.abs(narrow))) or 1.0
    narrow = _mulaw(narrow / peak * 0.9)
    return resample_poly(narrow, 2, 1).astype(np.float32)               # back to 16 kHz for ASR


class Recognizer:
    def __init__(self, model: str = "base.en", device: str = "cpu"):
        """Part 1 and 2 ran on CPU int8. Part 3 transcribes far more audio, so on a GPU it uses
        float16 with the same model, decoding settings and greedy search."""
        from faster_whisper import WhisperModel
        self.name = model
        if device == "cuda":
            self.model = WhisperModel(model, device="cuda", compute_type="float16")
        else:
            self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1,
                                            condition_on_previous_text=False,
                                            without_timestamps=True)
        return " ".join(segment.text for segment in segments).strip()


def load_audio(raw: bytes) -> np.ndarray:
    """Any WAV/FLAC/OGG byte string -> mono float32 at 16 kHz."""
    import io
    import soundfile as sf
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        from math import gcd
        g = gcd(int(rate), RATE)
        audio = resample_poly(audio, RATE // g, int(rate) // g).astype(np.float32)
    return audio


# Open-source TTS for Part 3 (runs on Linux/Kaggle, unlike macOS `say`): Piper voices, two US and
# two UK English, rotated per query exactly like the Part 1 voices.
PIPER_VOICES = ("en_US-lessac-medium", "en_US-amy-medium", "en_GB-alan-medium", "en_GB-jenny_dioco-medium")
_piper: dict = {}


def piper_voice_for(qid: str) -> str:
    return PIPER_VOICES[zlib.crc32(qid.encode()) % len(PIPER_VOICES)]


def synthesize_piper(text: str, voice: str) -> np.ndarray:
    """Text -> mono float32 at 16 kHz with a Piper voice (downloaded once from rhasspy/piper-voices)."""
    import io
    if voice not in _piper:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        lang, name, quality = voice.split("-")
        sub = f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}.onnx"
        onnx = hf_hub_download("rhasspy/piper-voices", sub)
        hf_hub_download("rhasspy/piper-voices", sub + ".json")
        _piper[voice] = PiperVoice.load(onnx)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        if hasattr(_piper[voice], "synthesize_wav"):
            _piper[voice].synthesize_wav(text, handle)
        else:                                   # piper-tts < 1.3
            _piper[voice].synthesize(text, handle)
    return load_audio(buf.getvalue())


_NORM = re.compile(r"[^a-z0-9' ]+")


def _words(text: str) -> list[str]:
    # Drop periods before splitting so "U.S." and "US" compare as the same single word.
    return _NORM.sub(" ", text.lower().replace(".", "")).split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


# Below this character-level similarity a no-overlap transcript is an invention, not a mishearing.
# Calibrated on NFCorpus telephone transcripts: "prunes" -> "I'm going to go to the bathroom"
# scores 0.07, while "cinnamon" -> "In a minute" (a genuine sound-alike error) scores 0.59.
HALLUCINATION_SIMILARITY = 0.35


def is_hallucination(reference: str, transcript: str) -> bool:
    """No word in common, longer than the query, and not even sound-alike at character level."""
    if set(_words(reference)) & set(_words(transcript)) or wer(reference, transcript) <= 1.0:
        return False
    squash = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())        # noqa: E731
    return difflib.SequenceMatcher(None, squash(reference), squash(transcript)).ratio() < HALLUCINATION_SIMILARITY
