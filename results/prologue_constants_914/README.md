# Initial work before the fixed PC tables

`pc_prologue` permits a straight-line initial prefix before address 14. A free
FLOW slot at the end jumps over the fixed tables, preserving absolute PCs and
dynamic cycles. The initial pause is scheduled in the graph with dependencies
before every STORE, so the pause-enabled harness observes unchanged memory.
Constant FLOW operations may now use cycle zero. Prefix length is derived from
each schedule; bundle counts are exact when operation times are supplied.

Twenty-two configurations vary scalar constants and the preceding 914 sources.
Two retained controls are fully verified on ten frozen seeds:

| Candidate | Cycles | Static bundles | W |
| --- | ---: | ---: | ---: |
| 004, earlier table layout | 913 | 10,795 | 54,484 |
| 012, interleaved tables | 913 | 11,043 | 54,427 |

Both retain every hash checkpoint, executed-PC mapping, output and non-output
memory, and match their standalone decoder. Candidate 012 leads to the promoted
913 checkpoint after measured constant transfers. The arithmetic-only minimum
over this screen is 54,415; it is not an achieved sub-900 schedule.

```sh
python3 search_prologue_constants.py
python3 export_candidate.py results/prologue_constants_914/candidate_012
```
