# Prefix layout with exactly preserved operation times

The interleaved 914 source's already verified pause is inserted into its graph.
Every other operation and scheduled time stays identical. Dependencies,
capacity, allocation and bootstrap placement are checked before export.

The resulting **914-cycle / 11,043-bundle** program passes ten frozen seeds and
standalone expansion equality. Its bootstrap is cycle 13 and pause cycle 1.
This isolates the thirteen-bundle layout saving from scheduling improvements.

```sh
python3 results/prologue_warm_914/run.py
python3 export_candidate.py results/prologue_warm_914
```
