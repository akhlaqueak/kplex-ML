import sys
from typing import Optional

from ._core import solve_args as _solve_args


def main() -> int:
    solve_args(sys.argv[1:], verbose=True)
    return 0


def solve_args(args: list[str], *, verbose: bool = False):
    return _solve_args(args, verbose)


def solve(
    dataset: Optional[str] = None,
    k: Optional[int] = None,
    *,
    graph_path: Optional[str] = None,
    algorithm: str = "kPEX",
    branching: Optional[str] = None,
    bounding: Optional[str] = None,
    ubr2: bool = True,
    ctcp: Optional[str] = None,
    heuristic: Optional[str] = None,
    framework: Optional[str] = None,
    verbose: bool = False,
):
    if dataset is None:
        dataset = graph_path

    if dataset is None:
        raise TypeError("solve() missing required argument: 'dataset'")
    if k is None:
        raise TypeError("solve() missing required argument: 'k'")

    args = [dataset, str(k)]

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

    return solve_args(args, verbose=verbose)
