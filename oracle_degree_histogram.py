#!/usr/bin/env python3
"""Create an aggregate vertex-degree histogram for every graph in an Oracle database."""

import argparse
import csv
import io
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    import torch_geometric  # Required when deserializing PyG Data objects.
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: torch_geometric. Install it with:\n"
        "  python3 -m pip install --user torch-geometric"
    ) from exc


DEFAULT_DB_PATH = "/home/akhlaque.ak@gmail.com/graphs/oracle"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the aggregate degree distribution of all Oracle graphs."
    )
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oracle-degree-histogram.png"),
        help="PNG histogram path (default: oracle-degree-histogram.png).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of the default logarithmic scale.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    degree_counts = Counter()
    total_vertices = 0
    with sqlite3.connect(db_path) as conn:
        total_graphs = conn.execute("SELECT COUNT(*) FROM rwd").fetchone()[0]
        cursor = conn.execute("SELECT data FROM rwd")
        for graph_index, (blob,) in enumerate(cursor, start=1):
            graph = torch.load(io.BytesIO(blob), weights_only=False)
            num_nodes = int(graph.num_nodes)
            degrees = torch.bincount(graph.edge_index[0], minlength=num_nodes)
            degree_counts.update(degrees.tolist())
            total_vertices += num_nodes

            if graph_index % 1000 == 0 or graph_index == total_graphs:
                print(f"Read {graph_index}/{total_graphs} graphs", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["degree", "vertex_count"])
        writer.writerows(sorted(degree_counts.items()))

    degrees = sorted(degree_counts)
    counts = [degree_counts[degree] for degree in degrees]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(degrees, counts, width=1.0, color="#2b6cb0", edgecolor="none")
    axis.set_xlabel("Vertex degree")
    axis.set_ylabel("Number of vertices")
    axis.set_title(
        f"Oracle Dataset Degree Distribution ({total_graphs:,} graphs; {total_vertices:,} vertices)"
    )
    if not args.linear_y:
        axis.set_yscale("log")
        axis.set_ylabel("Number of vertices (log scale)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200)

    print(f"Wrote histogram: {args.output}")
    print(f"Wrote counts: {csv_path}")


if __name__ == "__main__":
    main()
