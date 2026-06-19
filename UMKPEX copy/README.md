# umkpex

Python package wrapper for the `UMKPEX` solver.

## Install

```bash
cd UMKPEX
pip install -e .
```

## Use

```bash
umkpex path/to/graph.bin 5
python -m umkpex path/to/graph.bin 5 -algorithm UMKP+ -branching Pivot-Br
```

The wrapper runs the native `UMKPEX` binary. If the binary is missing, or older
than the local C++ sources, it rebuilds it with the same compile flags as the
project makefile.
