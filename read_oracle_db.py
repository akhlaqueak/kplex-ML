#!/usr/bin/env python3
"""Inspect and read BEACON Oracle Dataset SQLite files."""

import argparse
import io
import sqlite3
from pathlib import Path

import torch


DEFAULT_DB_PATH = "/home/akhlaque.ak@gmail.com/graphs/oracle"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read serialized PyG Data objects from a BEACON Oracle SQLite database."
    )
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=1, help="Number of sample graphs to print.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(rwd)").fetchall()
        if not columns:
            raise RuntimeError("Table 'rwd' was not found in this database.")

        print("rwd schema:")
        for _, name, column_type, *_ in columns:
            print(f"  {name}: {column_type}")

        total = conn.execute("SELECT COUNT(*) FROM rwd").fetchone()[0]
        print(f"\nGraphs: {total}")

        rows = conn.execute("SELECT data FROM rwd LIMIT ?", (args.limit,)).fetchall()

    for index, (blob,) in enumerate(rows):
        # The Oracle Dataset stores each graph as torch.save(Data(...)) bytes.
        graph = torch.load(io.BytesIO(blob), weights_only=False)
        print(f"\nGraph {index}:")
        print(graph)
        print(f"num_nodes: {graph.num_nodes}")
        print(f"edge_index shape: {tuple(graph.edge_index.shape)}")
        if hasattr(graph, "gt_induced_le5"):
            print(f"gt_induced_le5 shape: {tuple(graph.gt_induced_le5.shape)}")


if __name__ == "__main__":
    main()
