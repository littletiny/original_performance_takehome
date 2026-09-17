# Independent deep-level coordinate choices

`overfetch_by_level` selects groups separately for depths 8, 9 and 10. This
separates the FLOW savings of adjacent shifted coordinates from the scalar XOR
and scratch cost at each level. Unspecified levels use normal addresses. The
original `overfetch_groups` behavior is retained when this option is absent.

Two reference-seed regressions cover mixed normal/shifted transitions. The
24-case `search_overfetch_levels.py` screen uses 28 trials of 650 iterations;
every graph passes reference semantics first. No result beats production 914.

Candidate 011 is frozen-verified at **915 cycles / 10,808 bundles**, W=54,310.
It passes seeds 0–9, all retained hash checkpoints, PC mapping, output and
non-output memory checks, plus exact standalone expansion. Other reported
scores are allocated screens. The lowest window bound in the inventory is
907 and the lowest W is 54,286; neither is an achieved sub-900 schedule.
