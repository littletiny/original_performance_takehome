# Resource release/tail bounds

`resource_bounds.py` derives lower bounds for a fixed SSA graph, including its
fixed compound dispatch units. For each operation it computes an earliest
issue time and the minimum remaining tail from graph dependencies. If a set
of operations has release at least R, tail at least T, and work W on a resource
of capacity P, every complete schedule needs:

```text
cycles >= R + T + ceil(W / P)
```

The report scans all release/tail thresholds for each engine and for weighted
ALU/VALU capacity. It checks the result against each saved schedule and binds
the evidence to a graph digest. It ignores scratch constraints and the added
bootstrap/pause operations, so it can underestimate the actual minimum.

For the submitted 918-cycle graph, the strongest bound is **915 cycles**.
In the full grandchild-row probe the LOAD bound rises from 873 to 914 even
though its arithmetic count falls. These are fixed-graph bounds, not lower
bounds for alternative algorithms or for the overall take-home.

```sh
python3 resource_bounds.py results/compact_918 \
  results/grand_rows_918/candidate_025 \
  results/grand_rows_918/candidate_032 \
  results/depth4_918/candidate_027 \
  --output results/resource_windows_918/bounds.json
```
