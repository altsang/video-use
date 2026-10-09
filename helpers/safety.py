"""Refuse to write inside read-only footage folders.

VIDEO_USE_READONLY_DIRS (environment, or a .env at the repo root or in the current directory) lists
folders, separated by os.pathsep (";" on Windows, ":" elsewhere), that must never be written to -
e.g. the OneDrive folders holding original footage. Work on copies and point --edit-dir / -o elsewhere.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path


def env_value(key: str) -> str:
    """A setting from .env (repo root, then cwd), else the environment. Read as UTF-8 so Korean
    paths survive on Windows whatever the console code page is."""
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


def readonly_dirs() -> list[str]:
    raw = env_value("VIDEO_USE_READONLY_DIRS")
    return [d for d in raw.split(os.pathsep) if d.strip()]


def _norm(p: Path | str) -> str:
    """Comparable form of a path: resolved, Unicode-NFC (macOS stores Korean names decomposed,
    typed names are composed), and case-folded where the filesystem ignores case."""
    s = unicodedata.normalize("NFC", os.path.realpath(p))
    return s.casefold() if sys.platform in ("win32", "darwin") else s


def check_writable(path: Path | str) -> None:
    """Exit with a clear message if `path` is inside a read-only footage folder."""
    target = _norm(path)
    for d in readonly_dirs():
        root = _norm(d)
        if target == root or target.startswith(root.rstrip(os.sep) + os.sep):
            sys.exit(f"refusing to write inside a read-only footage folder ({d}): {path}\n"
                     "Copy the footage to a working folder first and write there.")
