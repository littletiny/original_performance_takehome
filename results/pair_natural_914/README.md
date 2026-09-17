# Natural PC layout without quartet selection

Twelve configurations apply `natural_pc_order` to early/late child pairs while
leaving unrelated handler execution in its original order. The best is
**916 cycles / 10,809 bundles**, W=54,470, in `candidate_000`; it does not beat
production 914.

Seeds 0–9 pass frozen execution, every retained hash checkpoint, PC mapping,
outputs and all non-output memory. The standalone decoder exactly matches the
verified program. The remaining candidates are allocated search results only.

```sh
python3 results/pair_natural_914/run.py
python3 export_candidate.py results/pair_natural_914/candidate_000
```
