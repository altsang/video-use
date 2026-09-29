"""Local Scribe stand-in: whisper-large-v3 on Apple Silicon via mlx_whisper.

Run by transcribe.py under the Python that has mlx_whisper installed
(WHISPER_PYTHON). Reads a 16 kHz mono wav, prints Scribe-shaped JSON to
stdout, so pack_transcripts / timeline_view / render work unchanged:

    {"language_code": "ko", "text": "...", "backend": "mlx-whisper",
     "words": [{"type": "word", "text": "...", "start": s, "end": e,
                "probability": p}, {"type": "spacing", ...}, ...]}

What Scribe gives that this does not: speaker_id (no diarization) and real
audio events. Instead, whisper's known failure modes are made visible:
  - looped segments ("딱 딱 딱 ...") are dropped;
  - segments matching YouTube-outro hallucinations are kept but preceded by an
    audio_event "(whisper: possible hallucination)" so they show up in
    takes_packed.md and never reach captions.

Settings follow video-director's indexer (tools/index/index_clip.py):
condition_on_previous_text=False, word timestamps, hallucination_silence_threshold,
highpass + loudnorm first (DJI Pocket audio sits near -36 dB; loudness does not
move timestamps), and a ko/en retry when language detection lands elsewhere.

Usage:
    <whisper-python> helpers/whisper_mlx_worker.py <audio.wav> [--language ko]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib

MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")
LANGS = ("ko", "en")  # anything else detected is treated as a misdetection

SUSPECT = ("이곳은 한국의 한", "한국의 한 수업", "시청해 주셔서", "시청해주셔서", "다음 영상에서", "구독",
           "좋아요와 알림", "자막", "MBC", "KBS", "SBS", "뉴스", "thank you for watching",
           "thanks for watching", "see you in the next", "subscribe")


def looped(text: str) -> bool:
    toks = text.split()
    if len(toks) >= 8 and max(toks.count(t) for t in set(toks)) / len(toks) > 0.5:
        return True
    raw = text.encode()
    return len(raw) > 120 and len(zlib.compress(raw)) / len(raw) < 0.25


def suspect(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in SUSPECT)


def run_whisper(wav: str, language: str | None) -> dict:
    import mlx_whisper
    kw = dict(path_or_hf_repo=MODEL, condition_on_previous_text=False,
              word_timestamps=True, hallucination_silence_threshold=1.0)
    if language:
        return mlx_whisper.transcribe(wav, language=language, **kw)
    r = mlx_whisper.transcribe(wav, **kw)
    if r.get("language") not in LANGS:
        tries = [mlx_whisper.transcribe(wav, language=lang, **kw) for lang in LANGS]
        conf = lambda x: (sum(s["avg_logprob"] for s in x["segments"]) / len(x["segments"])
                          if x["segments"] else -9)
        r = max(tries, key=conf)
    return r


def to_scribe(r: dict) -> dict:
    words: list[dict] = []
    texts = [s["text"].strip() for s in r["segments"]]
    dropped = 0
    for seg, text in zip(r["segments"], texts):
        if not text:
            continue
        if looped(text) or texts.count(text) >= 3:
            dropped += 1
            continue
        seg_words = [w for w in seg.get("words", []) if w["word"].strip()]
        if not seg_words:
            continue
        if suspect(text):
            words.append({"type": "audio_event", "text": "(whisper: possible hallucination)",
                          "start": round(seg_words[0]["start"], 3),
                          "end": round(seg_words[0]["start"], 3)})
        for w in seg_words:
            start, end = round(w["start"], 3), round(w["end"], 3)
            prev = next((x for x in reversed(words) if x["type"] == "word"), None)
            if prev is not None and start > prev["end"]:
                words.append({"type": "spacing", "text": " ", "start": prev["end"], "end": start})
            words.append({"type": "word", "text": w["word"].strip(), "start": start,
                          "end": max(end, start), "probability": round(w.get("probability", 0.0), 3)})
    return {
        "language_code": r.get("language"),
        "text": " ".join(w["text"] for w in words if w["type"] == "word"),
        "words": words,
        "backend": "mlx-whisper",
        "model": MODEL,
        "segments_dropped_as_loops": dropped,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--language", default=None)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        norm = os.path.join(tmp, "norm.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-i", args.wav, "-ac", "1", "-ar", "16000",
                        "-af", "highpass=f=80,loudnorm=I=-18", "-y", norm], check=True)
        r = run_whisper(norm, args.language)
    json.dump(to_scribe(r), sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
