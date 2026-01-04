# Taming the Performance Unpredictability of the Maximum $k$-Plex Finding Algorithms by Machine Learning

## To run a program
Go to particular folder you want to run

``` 
make
./run graph k
```
Sample graphs are provided in graphs

## ML Source Code
ML dataset and notebooks are available in ML_datasets_notebooks. Select particular model as given in the name of jupyter notebook. 


````md
# kplex-ML — Command Line Arguments

This program provides configurable algorithms, branching strategies, bounding techniques, heuristics, and reduction rules for solving k-plex / clique-style problems.  
All configuration is handled through command-line arguments parsed at startup.

---

## Usage

```bash
./program <dataset> <K> [options]
````

---

## Required Positional Arguments

| Argument    | Description                                    |
| ----------- | ---------------------------------------------- |
| `<dataset>` | Path or name of the input dataset              |
| `<K>`       | Integer parameter controlling the problem size |

---

## Optional Arguments

Optional arguments are provided as key–value pairs:

```bash
-option value
```

If an option is omitted, its default value is used.

---

## Algorithm Selection

### `-algorithm`

Specifies the main algorithm used by the solver.

* **Default:** `UMKP+`
* **Valid values:**

  * `UMKP`
  * `UMKP+`
  * `kPEX`

If an invalid value is provided, the program falls back to `kPEX`.

**Example:**

```bash
-algorithm kPEX
```

---

## Branching Strategy

### `-branching`

Selects the branching strategy used during the search.

* **Default:** `Default-Br`
* **Valid values:**

  * `Default-Br`
  * `Pivot-Br`
  * `S-Br`
  * `R-Br`
  * `SR-Br`

If an invalid value is provided, `Default-Br` is used.

**Example:**

```bash
-branching Pivot-Br
```

---

## Bounding Strategy

### `-bounding`

Controls the bounding method applied to prune the search space.

* **Default:** `None`
* **Valid values:**

  * `None`
  * `S-Bound`
  * `R-Bound`
  * `SR-Bound`
  * `AltRB`

If an invalid value is provided, the program falls back to `None`.

**Example:**

```bash
-bounding SR-Bound
```

---

## Reduction Rules

### `-UBR2`

Enables or disables the UBR2 reduction rule.

* **Default:** `true`
* **Accepted values:** `true`, `false`

**Example:**

```bash
-UBR2 false
```

---

## CTCP Method

### `-ctcp`

Specifies the CTCP method used by the algorithm.

* **Default:** `CF-CTCP`
* **Valid values:**

  * `CTCP`
  * `CF-CTCP`

If an invalid value is provided, the program falls back to `CF-CTCP`.

**Example:**

```bash
-ctcp CTCP
```

---

## Heuristic Selection

### `-huer`

Selects the heuristic applied during the computation.

* **Default:** `kPEX-Heur`
* **Valid values:**

  * `kPEX-Heur`
  * `EGO-Heur`
  * `Degen-Heur`

If an invalid value is provided, the program falls back to `kPEX-Heur`.

**Example:**

```bash
-huer EGO-Heur
```

---

## Example Command

```bash
./program dataset.txt 5 \
  -algorithm kPEX \
  -branching Pivot-Br \
  -bounding SR-Bound \
  -ctcp CF-CTCP \
  -huer Degen-Heur \
  -UBR2 true
```

---

## Startup Output

When the program starts, it prints a summary of the resolved parameters:

```
----------------------Parameters-----------------------
Dataset: dataset.txt
Algorithm: kPEX
K: 5
Branching: Pivot-Br
Bounding: SR-Bound
UBR2: true
CTCP Method: CF-CTCP
Heuristic: Degen-Heur
----------------------**********-----------------------
```

---

## Notes

* Invalid option values are automatically detected and replaced with defaults.
* Boolean options must be specified explicitly as `true` or `false`.
* Performance and runtime are strongly affected by the chosen algorithm, branching, bounding, and heuristic settings.

---

```
```
