# UMKP

Python package for the `UMKPEX` solver using `pybind11`.

## Install

```bash
cd UMKPEX
pip install -e .
```

## Use

```bash
UMKP path/to/graph.bin 5
python -m UMKP path/to/graph.bin 5 -algorithm UMKP+ -branching Pivot-Br
```

Or from Python:

```python
import UMKP

result = UMKP.solve("path/to/graph.bin", 5, algorithm="UMKP+", branching="Pivot-Br")
print(result["max_kplex_size"])
```
