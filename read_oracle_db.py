#!/usr/bin/env python3
"""Inspect and read BEACON Oracle Dataset SQLite files."""

import argparse
import io
import random
import sqlite3
import struct
from pathlib import Path

import torch

try:
    import torch_geometric  # Required by torch.load to reconstruct PyG Data objects.
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: torch_geometric. Install it with:\n"
        "  python3 -m pip install --user torch-geometric"
    ) from exc


DEFAULT_DB_PATH = Path("oracle")


def write_kplex_binary(graph, output_path: Path) -> tuple[int, int]:
    """Write a simple undirected graph in the CSR .bin format used by k-plex solvers."""
    num_nodes = int(graph.num_nodes)
    undirected_edges = set()
    for source, target in graph.edge_index.t().tolist():
        if source == target:
            continue
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"Edge ({source}, {target}) is outside [0, {num_nodes}).")
        undirected_edges.add((min(source, target), max(source, target)))

    adjacency = [[] for _ in range(num_nodes)]
    for source, target in undirected_edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    for neighbors in adjacency:
        neighbors.sort()

    degrees = [len(neighbors) for neighbors in adjacency]
    neighbors = [vertex for row in adjacency for vertex in row]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        # Header: sizeof(uint32_t), n, m (directed adjacency entries).
        output.write(struct.pack("<III", 4, num_nodes, len(neighbors)))
        output.write(struct.pack(f"<{num_nodes}I", *degrees))
        output.write(struct.pack(f"<{len(neighbors)}I", *neighbors))

    return len(undirected_edges), len(neighbors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read serialized PyG Data objects from a BEACON Oracle SQLite database."
    )
    parser.add_argument("db_path", nargs="?", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=1, help="Number of sample graphs to print.")
    parser.add_argument(
        "--min-nodes",
        type=int,
        help="List every graph with more than this many vertices instead of loading samples.",
    )
    parser.add_argument(
        "--rowid",
        type=int,
        help="Load and print the graph with this SQLite rowid.",
    )
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Print the graph's edge list as source target pairs.",
    )
    parser.add_argument(
        "--adjacency",
        action="store_true",
        help="Print the graph's adjacency list for every vertex.",
    )
    parser.add_argument(
        "--export-bin",
        type=Path,
        help="Write the selected --rowid graph in the .bin format used by the k-plex programs.",
    )
    parser.add_argument(
        "--export-all",
        type=Path,
        help="Write every Oracle graph as a .bin file into this directory.",
    )
    parser.add_argument(
        "--vertex-plot",
        "--histogram",
        dest="vertex_plot",
        nargs="?",
        type=Path,
        const=Path("oracle-vertex-counts.png"),
        help="Plot graph row number against vertex count (default: oracle-vertex-counts.png).",
    )
    parser.add_argument(
        "--export-sample",
        type=Path,
        help="Write a random sample of large Oracle graphs as .bin files into this directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of graphs written by --export-sample (default: 1000).",
    )
    parser.add_argument(
        "--sample-min-nodes",
        type=int,
        default=0,
        help="Only sample graphs with more than this many vertices (default: 0, all graphs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by --export-sample (default: 0).",
    )
    args = parser.parse_args()
    if args.export_bin is not None and args.rowid is None:
        parser.error("--export-bin requires --rowid.")
    if args.export_all is not None and args.rowid is not None:
        parser.error("--export-all cannot be combined with --rowid.")
    if args.export_all is not None and args.min_nodes is not None:
        parser.error("--export-all cannot be combined with --min-nodes.")
    if args.export_sample is not None and args.rowid is not None:
        parser.error("--export-sample cannot be combined with --rowid.")
    if args.export_sample is not None and args.min_nodes is not None:
        parser.error("--export-sample cannot be combined with --min-nodes.")
    if args.export_all is not None and args.export_sample is not None:
        parser.error("Use either --export-all or --export-sample, not both.")
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive.")

    db_path = Path(args.db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        columns = conn.execute("PRAGMA table_info(rwd)").fetchall()
        if not columns:
            raise RuntimeError("Table 'rwd' was not found in this database.")

        print("rwd schema:")
        column_names = {name for _, name, *_ in columns}
        for _, name, column_type, *_ in columns:
            print(f"  {name}: {column_type}")

        total = conn.execute("SELECT COUNT(*) FROM rwd").fetchone()[0]
        print(f"\nGraphs: {total}")

        if args.vertex_plot is not None:
            try:
                import matplotlib.pyplot as plt
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    "Plot support requires matplotlib. Install it with:\n"
                    "  python3 -m pip install --user matplotlib"
                ) from exc

            graph_ids = []
            vertex_counts = []
            edge_counts = []
            cursor = conn.execute(
                "SELECT rowid, num_nodes, data FROM rwd WHERE num_nodes > 0 ORDER BY rowid"
            )
            for index, (row_id, num_nodes, blob) in enumerate(cursor, start=1):
                graph = torch.load(io.BytesIO(blob), weights_only=False)
                graph_ids.append(row_id)
                vertex_counts.append(num_nodes)
                # Oracle's undirected graphs store each edge in both directions.
                edge_counts.append(graph.edge_index.size(1) // 2)
                if index % 1000 == 0 or index == total:
                    print(f"Read {index}/{total} graphs for the vertex/edge plot", flush=True)

            args.vertex_plot.parent.mkdir(parents=True, exist_ok=True)
            fig, axis = plt.subplots(figsize=(10, 5.5))
            edge_axis = axis.twinx()
            vertex_plot = axis.scatter(
                graph_ids, vertex_counts, s=3, alpha=0.55, color="#3B6EA5", linewidths=0,
                label="Vertices",
            )
            edge_plot = edge_axis.scatter(
                graph_ids, edge_counts, s=3, alpha=0.40, color="#D97706", linewidths=0,
                label="Undirected edges",
            )
            axis.set_xlabel("Graph number (SQLite rowid)")
            axis.set_ylabel("Number of vertices", color="#3B6EA5")
            edge_axis.set_ylabel("Number of undirected edges", color="#D97706")
            axis.tick_params(axis="y", labelcolor="#3B6EA5")
            edge_axis.tick_params(axis="y", labelcolor="#D97706")
            axis.set_title(f"Oracle Dataset Graph Sizes (n={len(vertex_counts):,})")
            axis.legend(handles=[vertex_plot, edge_plot], loc="upper left")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.vertex_plot, dpi=200)
            plt.close(fig)
            print(f"Wrote vertex-count plot to {args.vertex_plot}")
            return

        if args.export_all is not None:
            args.export_all.mkdir(parents=True, exist_ok=True)
            manifest_path = args.export_all / "manifest.tsv"
            with manifest_path.open("w", encoding="utf-8") as manifest:
                manifest.write("rowid\tfile\tnum_nodes\tundirected_edges\n")
                cursor = conn.execute("SELECT rowid, data FROM rwd ORDER BY rowid")
                for index, (row_id, blob) in enumerate(cursor, start=1):
                    graph = torch.load(io.BytesIO(blob), weights_only=False)
                    output_path = args.export_all / f"oracle_{row_id}.bin"
                    edge_count, _ = write_kplex_binary(graph, output_path)
                    manifest.write(
                        f"{row_id}\t{output_path.name}\t{graph.num_nodes}\t{edge_count}\n"
                    )
                    if index % 1000 == 0 or index == total:
                        print(f"Exported {index}/{total} graphs")
            print(f"Wrote .bin files to {args.export_all}")
            print(f"Wrote manifest to {manifest_path}")
            return

        if args.export_sample is not None:
            candidate_ids = [
                row_id
                for (row_id,) in conn.execute(
                    "SELECT rowid FROM rwd WHERE num_nodes > ?", (args.sample_min_nodes,)
                )
            ]
            if len(candidate_ids) < args.sample_size:
                raise ValueError(
                    f"Only {len(candidate_ids)} graphs have more than "
                    f"{args.sample_min_nodes} nodes; cannot select {args.sample_size}."
                )

            selected_ids = random.Random(args.seed).sample(candidate_ids, args.sample_size)
            args.export_sample.mkdir(parents=True, exist_ok=True)
            manifest_path = args.export_sample / "manifest.tsv"
            with manifest_path.open("w", encoding="utf-8") as manifest:
                manifest.write("rowid\tfile\tnum_nodes\tundirected_edges\n")
                for index, row_id in enumerate(selected_ids, start=1):
                    blob = conn.execute(
                        "SELECT data FROM rwd WHERE rowid = ?", (row_id,)
                    ).fetchone()[0]
                    graph = torch.load(io.BytesIO(blob), weights_only=False)
                    output_path = args.export_sample / f"oracle_{row_id}.bin"
                    edge_count, _ = write_kplex_binary(graph, output_path)
                    manifest.write(
                        f"{row_id}\t{output_path.name}\t{graph.num_nodes}\t{edge_count}\n"
                    )
                    if index % 100 == 0 or index == args.sample_size:
                        print(f"Exported {index}/{args.sample_size} sampled graphs")
            print(
                f"Wrote {args.sample_size} graphs with more than {args.sample_min_nodes} nodes "
                f"to {args.export_sample} (seed={args.seed})"
            )
            print(f"Wrote manifest to {manifest_path}")
            return

        if args.min_nodes is not None:
            selected_columns = "rowid, num_nodes"
            if "domain" in column_names:
                selected_columns += ", domain"
            rows = conn.execute(
                f"SELECT {selected_columns} FROM rwd "
                "WHERE num_nodes > ? ORDER BY num_nodes, rowid",
                (args.min_nodes,),
            ).fetchall()
            print(f"Graphs with more than {args.min_nodes} nodes: {len(rows)}")
            for row in rows:
                row_id, num_nodes = row[:2]
                domain = row[2] if len(row) > 2 else "n/a"
                print(f"rowid={row_id} num_nodes={num_nodes} domain={domain}")
            return

        if args.rowid is not None:
            rows = conn.execute(
                "SELECT rowid, data FROM rwd WHERE rowid = ?", (args.rowid,)
            ).fetchall()
            if not rows:
                raise ValueError(f"No graph found with rowid={args.rowid}.")
        else:
            rows = conn.execute("SELECT rowid, data FROM rwd LIMIT ?", (args.limit,)).fetchall()

    for index, (row_id, blob) in enumerate(rows):
        # The Oracle Dataset stores each graph as torch.save(Data(...)) bytes.
        graph = torch.load(io.BytesIO(blob), weights_only=False)
        print(f"\nGraph {index} (rowid={row_id}):")
        print(graph)
        print(f"num_nodes: {graph.num_nodes}")
        print(f"edge_index shape: {tuple(graph.edge_index.shape)}")
        if hasattr(graph, "gt_induced_le5"):
            print(f"gt_induced_le5 shape: {tuple(graph.gt_induced_le5.shape)}")

        edge_pairs = graph.edge_index.t().tolist()
        if args.edges:
            print("edge list:")
            for source, target in edge_pairs:
                print(f"{source} {target}")

        if args.adjacency:
            adjacency = [[] for _ in range(graph.num_nodes)]
            for source, target in edge_pairs:
                adjacency[source].append(target)
            print("adjacency list:")
            for vertex, neighbors in enumerate(adjacency):
                print(f"{vertex}: {' '.join(map(str, neighbors))}")

        if args.export_bin is not None:
            undirected_count, directed_count = write_kplex_binary(graph, args.export_bin)
            print(
                f"wrote {args.export_bin}: n={graph.num_nodes} "
                f"undirected_edges={undirected_count} directed_entries={directed_count}"
            )


if __name__ == "__main__":
    main()
