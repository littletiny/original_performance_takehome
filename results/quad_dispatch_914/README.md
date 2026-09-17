# Packed runtime quartets and base-four dispatch

This experiment replaces 32 scalar grandchild copies per selected group with
eight ordered VSTOREs and four VLOADs. Another VSTORE in each producer bundle
retains the child pair for round 4. The 64-word temporary row contains the
interleaved child pairs, eight contiguous quartets, and all overwritten padding.
Every temporary word is cleared. The new research-only `lookup_quad_store`
lowers to the unchanged machine's ordinary VSTORE.

Round 5 selects the runtime quartets through a shared base-four dispatch table.
Two groups use 16 cases per lane; three use 64. The optional folded path keeps
q3, forms `s = 2*b3+b4`, then reconstructs `q5 = 4*q3+s`. It preserves the normal
path arithmetic count. New PC setup, vector loads and synchronization consume
part of the scalar-copy savings.

Forty configurations vary selected groups, retained tree levels and deep
overfetch. Fifteen selected graphs pass both reference seeds before scheduling.
The best retained schedule is **916 cycles / 10,929 static bundles**, from
`candidate_000`. Its W is **54,455**, compared with production's 54,484; its
fixed-graph bound remains 910. This does **not** beat production 914.

The 916-cycle program passes frozen-machine seeds 0–9: all 20,480 retained hash
checkpoints per seed, executed-PC mapping, outputs and every non-output memory
word. Its standalone decoder expands to exactly the verified program. Source
padding is allocated even where its contents are ignored; the semantic checker
poisons those unused writes to test later overlap handling.

Further controls are retained separately:

- `../quad_natural_914/`: 16 execution-order/PC-layout controls; verified
  three-group selection at 917 / 11,314.
- `../quad_joint_914/`: 192 combined graphs reach arithmetic bound 898 and
  best resource-window bound 902, on different graphs. No selected large graph
  retains an allocated schedule; measured scratch pressure explains three
  diagnostic failures.
- `../quad_integration_914/`: verified 936 / 11,333 with quartet rows, early and
  late child rows, natural PC layout and chained producers together.
- `../pair_natural_914/`: 12 controls apply natural PC layout without quartets.

The production source and its saved SSA graph are unchanged. The static cap is
12,000; the **<900-cycle goal remains unmet**. Regression results are retained
in `../quad_regression_tests.log`.

```sh
python3 search_quad_dispatch.py
python3 export_candidate.py results/quad_dispatch_914/candidate_000
python3 -m unittest test_quad_rows test_schedule -v
```
