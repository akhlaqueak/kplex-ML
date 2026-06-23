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

From Python:

```python
import struct

result = struct.solve(dataset="path/to/graph.bin", k=5, algorithm="UMKP+", branching="Pivot-Br")
```
