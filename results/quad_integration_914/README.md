# Combined layout correctness control

The saved **936-cycle / 11,333-bundle** program combines a three-group quartet
dispatch, two quartet buffers, chained quartet producers, early and late child
rows, reversed custom late-row order and natural physical PC layout.

Its W is 54,403 and scratch live peak is 1,431. Frozen seeds 0–9 pass all 20,480
retained hash checkpoints, executed-PC mappings, outputs and non-output memory.
The standalone decoder is identical to the verified program. This is a
correctness control, not a production promotion or a sub-900 result.

```sh
python3 results/quad_integration_914/run.py
python3 export_candidate.py results/quad_integration_914/candidate_000
```
