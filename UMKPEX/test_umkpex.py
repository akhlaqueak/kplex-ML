#!/usr/bin/env python3

import json

import UMKP


def main() -> int:
    result = UMKP.solve(
        "ds/hood.bin",
        7,
        algorithm="UMKP+",
        branching="Pivot-Br",
        bounding="AltRB",
        ctcp="CTCP",
        heuristic="Both-Heur",
        framework="default",
        ubr2=True,
    )

    print("Summary:")
    print(json.dumps(result["summary"], indent=2))
    print("\nSolution size:", result["MaxKPlex_Size"])
    print("Solution vertices:", result["solution_vertices"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
