# Constant rewrites respecting an existing schedule

`search_scheduled_constants.py` considers exact scalar-constant formulas only
when both operands are already available at the target's issue time in the
source schedule. It scores readiness or lifetime extension and tries budgets
of 8, 32 or all improvements on production 915 and its early-prefix control.
Each explicit formula is independently checked modulo 2^32 in the builder.

For all 12 variants, the saved source times are checked against every graph
dependency and every resource capacity. Scratch allocation is checked
separately. A feasible warm schedule is retained if subsequent heuristic
search cannot improve it; both the warm and searched results are explicit.
Each graph receives 32 native trials of 600 iterations.

Candidate 008 is frozen-verified at **914 cycles / 10,807 bundles**, with
98 constant-expression replacements and early address prefixes for groups
0 and 1. It has W=54,592 and a scratch live-word peak of 1,356. Seeds 0–9 pass
all 20,480 hash checkpoints, PC mappings, output and non-output memory checks,
and the standalone program expansion matches exactly.

The other variants retain 915. The subsequent 24-case port-transfer screen
keeps 914 and reduces W to 54,536; that is the promoted `../compact_914/`.
The production builder is self-contained and does not run either search or
read any expression/configuration artifact at runtime.

```sh
python3 search_scheduled_constants.py
python3 export_candidate.py results/scheduled_constants_915/candidate_008
```
