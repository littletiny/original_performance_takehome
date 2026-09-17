# Scalar-constant dependency resynthesis

`constant_graph.py` considers exact 32-bit addition, subtraction and XOR among
known scalar constants. It rewrites existing ALU destinations, introduces no
instructions, and uses LOAD constants or known words without mutable-constant
ancestors as roots. Every chosen expression increases final depth, including
retained original expressions; this prevents cycles through forward references.
Runtime tree/input values are not included as numeric constants.

The real-machine regression covers forward references, wraparound and all
produced constant values. The full-graph regression checks runtime header
anchors and both reference seeds. `resynthesize_constants` enables the pass;
it is disabled in the promoted kernel's source configuration.

Six graph families receive 40 native trials of 800 iterations. The best
allocated result is **921**, frozen-verified on seeds 0–9 with 20,480 retained
hash checkpoints per seed, all PC mappings, outputs, non-output memory and
standalone expansion equality. Its artifact is candidate 001. Other retained
screens are 925 and 932; three graphs retain no allocated schedule.

This reduces some dependency depths but ignores when those operands actually
fit in a good resource schedule. The subsequent experiment in
`../scheduled_constants_915/` restricts choices using actual source times and
does improve the measured cycle count.

```sh
python3 test_constant_graph.py
python3 search_constant_dags.py
python3 export_candidate.py results/constant_dags_915/candidate_001
```
