# Retain the schedule while moving hash packs into holes

Eight contraction/expansion combinations retain 914 cycles. Candidate 000 moves
`r0.g19.h6.b` from eight ALU operations into one available VALU slot at cycle 166.
Its W remains 54,484, but the fixed-graph resource-window bound falls to 910.
It is fully frozen-verified on seeds 0–9 and promoted as `../compact_914_pc/`.

```sh
python3 rebalance_holes.py results/pc_delta_memory_914/candidate_000 results/pc_delta_holes_914 --moves '[[1,0],[2,0],[4,0],[8,0],[1,1],[2,2],[4,4],[8,8]]'
```
