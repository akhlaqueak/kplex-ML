# struct

Python package for the `UMKPEX` solver using `pybind11`.

## Install

```bash
cd UMKPEX
pip install -e .
```

## Use

```bash
struct path/to/graph.bin 5
python -m struct path/to/graph.bin 5 -algorithm UMKP+ -branching Pivot-Br
```

Available binary branching strategies:

- `Binary-Br`: kPlexT branching with its reduction rules applied before branching.
- `BinaryM-Br`: the previous minimum-degree binary branching strategy.

From Python:

```python
import struct

result = struct.solve(dataset="path/to/graph.bin", k=5, algorithm="UMKP+", branching="Pivot-Br")
```
