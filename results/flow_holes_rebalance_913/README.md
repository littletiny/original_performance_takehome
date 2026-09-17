# Balance arithmetic after constant rematerialization

Six measured-hole variants retain 913 cycles. `candidate_000` moves the root
mix `r0.g4.mix` from one VALU slot to eight ALU slots in cycle 5, preserving
weighted work at 54,414 while improving ALU/VALU count balance. Its source is
`../flow_holes_913/candidate_004`; the final LOAD-transfer descendant is verified
in `../compact_913_packed/`.

```sh
python3 rebalance_holes.py results/flow_holes_913/candidate_004 results/flow_holes_rebalance_913 --moves '[[0,1],[0,2],[0,4],[1,1],[2,2],[4,4]]'
```
