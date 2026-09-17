# Sub-900 requires room above a useful lower bound

For an integer cycle count, the mathematical requirement is `L <= T <= 899`.
It does not require `L < 899`: an exact 899-cycle schedule could attain its
899-cycle bound. The practical design target is lower because the calculated
bound omits coupled resource contention and scratch allocation.

| Fixed-graph lower bound L | Largest permitted T−L at 899 |
| --- | ---: |
| 899 | 0 |
| 895 | 4 |
| 890 | 9 |

Thus **890–895 is an engineering target, not a guarantee or a theorem**.
Measure the gap on each graph; do not assume a gap measured on one graph
survives substantial rewrites.

| Graph | Resource-window bound | Best frozen-verified schedule | Observed gap | Static bundles |
| --- | ---: | ---: | ---: | ---: |
| Production `compact_915` | 912 | 915 | 3 | 10,808 |
| Combined `bound899_915/candidate_009` | 899 | 967 | 68 | 11,900 |

The combined graph has fewer arithmetic operations, but its tested schedules
are much worse. The 68-cycle gap includes the limitations of heuristic search;
it is not a proof that this graph cannot be scheduled faster. Both rows passed
all 20,480 retained hash checkpoints per seed on seeds 0–9, PC mapping, output,
non-output memory and standalone expansion checks. Neither row achieves <900.

The production graph has `W = ALU + 8*VALU = 54,577`, with at most 60 W issued
per cycle. Count-only necessary cuts are:

| Pure arithmetic capacity target | W ceiling | Necessary cut from production |
| --- | ---: | ---: |
| 899 | 53,940 | 637 |
| 895 | 53,700 | 877 |
| 890 | 53,400 | 1,177 |

These are optimistic minimum reductions (approximately 1.6–2.2% for 890–895).
They do not establish the stronger resource-window target: initial readiness,
tail work, individual ALU/VALU balance, LOAD, FLOW and scratch must also fit.
Changing work onto another port is useful only if that port and its timing
windows have enough capacity.

This experiment adds several bounded ways to test that distinction:

- `../joint_inventory_915/`: 240 combinations of known reductions; best
  retained resource-window bound 902.
- `../cache_constant_trade_915/`: explicit cache/constant/chain configurations
  leading to the first 899-bound graph. Every configuration can be rebuilt.
- `../bound899_915/`: 60 variants screened, 11 with bound <=899 scheduled;
  best retained frozen result 967. No production promotion.
- `../uniform_operands_915/`: 16 scalar-uniform-operand screens; none beat 915.
- `../prefetch_pairs_915/`: early address and adjacent-child prefetch trials,
  including real-write and missing-order negative controls.

Rebuild `analysis.json` with `python3 results/sub900_margin_915/run.py`.
The submitted source remains the verified 915-cycle checkpoint.
