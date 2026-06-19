from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent
BINARY_PATH = PROJECT_DIR / "UMKPEX"
SOURCE_FILES = (
    "main.cpp",
    "Graph.h",
    "Branch.h",
    "KPlex_BB_matrix.h",
    "2th-Reduction.h",
    "CTPrune.h",
    "Command.h",
    "LinearHeap.h",
    "MyBitset.h",
    "Utility.h",
)


def _source_paths() -> list[Path]:
    return [PROJECT_DIR / name for name in SOURCE_FILES]


def binary_is_stale() -> bool:
    if not BINARY_PATH.exists():
        return True

    binary_mtime = BINARY_PATH.stat().st_mtime
    for path in _source_paths():
        if path.exists() and path.stat().st_mtime > binary_mtime:
            return True
    return False


def build_binary(force: bool = False) -> Path:
    if not force and not binary_is_stale():
        return BINARY_PATH

    cmd = [
        "g++",
        "-std=c++11",
        "-O3",
        "-w",
        "main.cpp",
        "-o",
        str(BINARY_PATH),
        "-DNO_PROGRESS_BAR",
        "-DNDEBUG",
    ]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
    return BINARY_PATH


def ensure_binary() -> Path:
    return build_binary(force=False)


def run_umkpex(
    args: Sequence[str] | None = None,
    *,
    check: bool = False,
    capture_output: bool = False,
    text: bool = True,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    binary = ensure_binary()
    cmd = [str(binary), *(args or [])]
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        cwd=cwd,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    result = run_umkpex(args, check=False, capture_output=False, text=True)
    return int(result.returncode)
