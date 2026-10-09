"""Transcribe a video with ElevenLabs Scribe.

Extracts mono 16kHz audio via ffmpeg, uploads to Scribe with verbatim +
diarize + audio events + word-level timestamps, writes the full response
to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the upload is skipped.

Backends (VIDEO_USE_ASR in .env or the environment):
    (unset)  ElevenLabs Scribe - uploads audio; diarization + audio events
    mlx      local Whisper large-v3 on Apple Silicon (WHISPER_PYTHON with mlx_whisper)
    faster   local faster-whisper on the CPU (Windows/Intel/any): WHISPER_MODEL (default medium),
             WHISPER_DOWNLOAD_ROOT (model cache), WHISPER_THREADS. No diarization/audio events.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # helpers/ even when loaded by file path
from safety import check_writable  # noqa: E402


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def env_value(key: str) -> str:
    """A setting from .env (repo root, then cwd), else the environment."""
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return os.environ.get(key, "")


def asr_backend() -> str:
    """'mlx' (Whisper on Apple Silicon), 'faster' (faster-whisper on the CPU), or '' (ElevenLabs Scribe)."""
    return env_value("VIDEO_USE_ASR").strip().lower()


def use_local_asr() -> bool:
    """VIDEO_USE_ASR=mlx or faster: transcribe on this machine - nothing uploaded."""
    return asr_backend() in ("mlx", "faster")


def load_api_key() -> str:
    if use_local_asr():
        return ""
    v = env_value("ELEVENLABS_API_KEY")
    if not v:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment (or set VIDEO_USE_ASR=mlx to transcribe locally)")
    return v


WHISPER_SCRIPT = r"""
import json, sys, mlx_whisper
audio, lang, model = sys.argv[1], sys.argv[2] or None, sys.argv[3]
r = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language=lang, word_timestamps=True,
                           condition_on_previous_text=False)
json.dump(r, sys.stdout, ensure_ascii=False)
"""


def call_whisper(audio_path: Path, language: str | None = None) -> dict:
    """Local Whisper in the Python named by WHISPER_PYTHON (needs mlx_whisper), reshaped to Scribe's
    response: words with type word / spacing, one speaker (no diarization, no audio events)."""
    py = env_value("WHISPER_PYTHON") or sys.executable
    model = env_value("WHISPER_MODEL") or "mlx-community/whisper-large-v3-mlx"
    out = subprocess.run([py, "-c", WHISPER_SCRIPT, str(audio_path), language or "", model],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"local whisper failed: {out.stderr[-800:]}")
    r = json.loads(out.stdout)
    words: list[dict] = []
    for seg in r.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"].strip()
            if not text:
                continue
            if words:
                words.append({"text": " ", "start": words[-1]["end"], "end": round(w["start"], 3),
                              "type": "spacing", "speaker_id": "speaker_0"})
            words.append({"text": text, "start": round(w["start"], 3), "end": round(w["end"], 3),
                          "type": "word", "speaker_id": "speaker_0",
                          "logprob": round(math.log(max(w.get("probability") or 1e-9, 1e-9)), 4)})
    return {"language_code": r.get("language"), "text": r.get("text", "").strip(), "words": words,
            "asr": f"local:{model}"}


def scribe_shape(word_iter, language: str | None, model: str) -> dict:
    """(text, start, end, probability) tuples -> Scribe's words/spacing shape, one speaker."""
    words: list[dict] = []
    for text, start, end, prob in word_iter:
        text = text.strip()
        if not text:
            continue
        if words:
            words.append({"text": " ", "start": words[-1]["end"], "end": round(start, 3),
                          "type": "spacing", "speaker_id": "speaker_0"})
        words.append({"text": text, "start": round(start, 3), "end": round(end, 3),
                      "type": "word", "speaker_id": "speaker_0",
                      "logprob": round(math.log(max(prob or 1e-9, 1e-9)), 4)})
    return {"language_code": language, "text": " ".join(w["text"] for w in words if w["type"] == "word"),
            "words": words, "asr": model}


def call_faster_whisper(audio_path: Path, language: str | None = None) -> dict:
    """faster-whisper on the CPU (int8) - for Windows/Intel machines without MLX. Nothing uploaded."""
    from faster_whisper import WhisperModel  # noqa: PLC0415 - optional dependency

    model_name = env_value("WHISPER_MODEL") or "medium"
    root = env_value("WHISPER_DOWNLOAD_ROOT") or None
    threads = int(env_value("WHISPER_THREADS") or 0)
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=root, cpu_threads=threads)
    segments, info = model.transcribe(str(audio_path), language=language or None, word_timestamps=True,
                                      condition_on_previous_text=False, beam_size=5)
    tuples = ((w.word, w.start, w.end, w.probability) for seg in segments for w in (seg.words or []))
    return scribe_shape(tuples, info.language, f"local:faster-whisper:{model_name}")


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    check_writable(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        # Uploading silence costs the same as uploading speech and returns
        # nothing, so catch the wrong-track case before paying for it (locally: before wasting time).
        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - not uploading. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        size_mb = audio.stat().st_size / (1024 * 1024)
        if use_local_asr():
            if verbose:
                print(f"  transcribing {video.stem}.wav locally ({size_mb:.1f} MB, no upload)", flush=True)
            payload = (call_faster_whisper(audio, language) if asr_backend() == "faster"
                       else call_whisper(audio, language))
        else:
            if verbose:
                print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            payload = call_scribe(audio, api_key, language, num_speakers)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with ElevenLabs Scribe")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
    )


if __name__ == "__main__":
    main()
