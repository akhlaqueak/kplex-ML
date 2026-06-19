from __future__ import annotations

import sys

from .runner import main as runner_main


def main() -> int:
    return runner_main(sys.argv[1:])
