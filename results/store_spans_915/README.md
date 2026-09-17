# STORE-copy spans with fixed PC pools and late child rows

Previously the builder rejected any combination of extended handler spans
with fixed PC pools or late paired-child rows. The actual layout permits them:
the PC pools serve only span-one singleton tables, wider tables follow them,
and late child rows remain span one. The input-backed extended buffer now has
an ordering key distinct from the index-backed child rows.

The fixed-table machine test covers both alternatives in span-one, span-two
and span-three handlers. The child-row regression checks separate buffer keys
and full reference semantics when both layouts coexist.

`search_store_spans.py` screens **192** configurations spanning six/eleven
grandchild caches, two/three-cycle cases, split early pairs, retained originals,
memory broadcasts, PC bit pools and dispatch-chain budgets. All 16 selected
graphs pass reference seeds 0 and 1 before 32 native trials of 600 iterations.

The best retained fixed-graph bound is **901**, W=53,844, and a hypothetical
899-cycle schedule would use 11,688 static bundles. The LOAD release/tail
window prevents that graph from reaching 899 even without scratch. None of
the 16 screens retains an allocated schedule; `100000` is a sentinel.

Two additional diagnostic schedules have:

| Candidate | Unallocated cycles | Static bundles | Live scratch words |
| --- | ---: | ---: | ---: |
| 164 | 974 | 11,763 | 1,604 |
| 172 | 976 | 11,765 | 1,673 |

Both exceed the real 1,536-word limit. These samples diagnose allocation
pressure; they do not prove every schedule for these graphs is infeasible.
See `allocation_diagnostics.json`. The broader screen is `screen.json`.

Smaller changes are measured separately in `../store_spans_local_915/`.
No extended STORE span is enabled in production 914.
