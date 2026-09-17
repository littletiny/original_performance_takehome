# Joint quartet and child-row reductions

The 192-case inventory varies eight/eleven quartet groups, retained tree levels,
chain length, quartet and broadcast buffer counts, initial broadcasts and FLOW
targets. Every inventory entry has a valid graph. Twelve selected graphs pass
both semantic seeds, but none retains a scratch-allocated schedule within the
search and 12,000-bundle limit. `100000` is a failure sentinel, not a cycle score.

| Control | W | Arithmetic bound | Resource-window bound |
| --- | ---: | ---: | ---: |
| Minimum work, candidate 129 | 53,850 | 898 | 923 |
| Best window bound, candidate 123 | 53,924 | 899 | 902 |

Candidate 129 is rebuilt from its recorded parameters and independently passes
both semantic seeds. Neither row is a frozen-machine performance result.
These bounds describe their fixed graphs and omit scratch constraints.

Three independent diagnostic schedules expose the allocation failure:

| Candidate | Cycles before allocation | Live words | Limit |
| --- | ---: | ---: | ---: |
| 123 | 987 | 1,623 | 1,536 |
| 19 | 988 | 1,666 | 1,536 |
| 96 | 989 | 1,793 | 1,536 |

See `allocation_diagnostic.json`; these are invalid schedules, not measured
kernel scores. They do not prove that every schedule of those graphs must fail.
A producer-first chain-order control is in `../quad_producers_first_914/`; its
window bound is 903, not an improvement over 902.

The remaining problem is to retain the arithmetic savings while reducing LOAD
release delays, coupled group readiness and scratch lifetimes. Merely reaching
an arithmetic bound below 899 is insufficient. Production remains 914.

```sh
python3 search_quad_joint.py
python3 results/quad_joint_914/diagnose.py
```
