# A 899 lower bound did not yield a sub-900 schedule

`search_bound899.py` screens 60 combinations of scalar hash-constant sharing,
scalar root mixes, retention of depth-6 originals and LOAD pointer anchors.
Eleven graphs passed the fixed-graph resource-window bound <=899. Each passed
two-seed semantic verification before scheduling. Each accepted graph received
60 native trials of 800 iterations with scratch allocation and the 12,000
static-bundle limit enforced.

The retained allocated schedules range from 967 to 999 cycles. Six variants
did not retain an allocated schedule; `100000` in the search summary denotes
that absence, not a measured cycle count. The full screen and summary are
`screen.json` and `summary.json`.

Candidate 009 is independently frozen-verified at **967 cycles / 11,900
bundles**, with bound **899**, W=53,825 and a scratch live-word peak of 1,318.
Seeds 0–9 pass every retained hash checkpoint, executed-PC mapping, outputs
and preservation of non-output memory. The standalone expansion equals the
verified lowerer's program. See `candidate_009/verification.json` and
`candidate_009/export.log`.

The observed 68-cycle difference is not a proven minimum overhead. The lower
bound ignores scratch and coupled resource contention, and the heuristic
search is not an optimality proof. It does show that a lower-bound filter alone
does not predict an improvement. The production kernel remains at 915 cycles.

```sh
python3 search_bound899.py
python3 export_candidate.py results/bound899_915/candidate_009
python3 results/sub900_margin_915/run.py
```
