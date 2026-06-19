import sys
from typing import Optional

from ._core import solve_args as _solve_args


def main() -> int:
    solve_args(sys.argv[1:])
    return 0


def solve_args(args: list[str]):
    return _solve_args(args)


def solve(
    graph_path: str,
    k: int,
    *,
    algorithm: str = "kPEX",
    branching: Optional[str] = None,
    bounding: Optional[str] = None,
    ubr2: bool = True,
    ctcp: Optional[str] = None,
    heuristic: Optional[str] = None,
    framework: Optional[str] = None,
):
    args = [graph_path, str(k)]

    if algorithm is not None:
        args.extend(["-algorithm", algorithm])
    if branching is not None:
        args.extend(["-branching", branching])
    if bounding is not None:
        args.extend(["-bounding", bounding])

    args.extend(["-UBR2", "true" if ubr2 else "false"])

    if ctcp is not None:
        args.extend(["-ctcp", ctcp])
    if heuristic is not None:
        args.extend(["-heuristic", heuristic])
    if framework is not None:
        args.extend(["-framework", framework])

    return solve_args(args)
