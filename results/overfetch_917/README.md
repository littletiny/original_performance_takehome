# Deep address recurrence using VLOAD's lane offset

In the unmodified deep-tree layout, the actual node address A is the one-based
heap coordinate q plus six. Its child address is `2*A - 6 + parity`. Keeping
`q = A - 6` instead makes the recurrence `q_next = 2*q + parity`.

For selected groups at depths 8–10, the implementation performs one real
VLOAD at q and consumes only lane six, which is the required node at A. This
replaces a scalar LOAD without adding a LOAD instruction. The transition from
the mirrored depth-seven layout still performs its required compensation.
The subsequent depth-eight and depth-nine updates each lose one FLOW select.

## Correctness and allocation

Every fetch owns an eight-word virtual value and reserves all eight physical
writes. The padding audit rejects any logical consumer/checkpoint of the seven
other lanes. Scalar node-mix XORs consume the useful lanes directly into the
ordinary contiguous hash-input vector, with no extra weighted mix work. Those
packs are excluded from automatic ALU/VALU migration.

Coordinates range from 256 to 2047; even the last VLOAD ends at word 2054,
inside the existing memory image. Padding may overlap unrelated tree/index
data, but it is unobserved. All node values remain runtime data and no machine
rule or memory length is changed. The semantic regression checks the full
recurrence and rejects an added padding consumer.

## Measurements

The 30-configuration screen uses `memory_vector_followup_917/candidate_007`
as its source. Every graph passed seeds 0 and 1 before scheduling. It varies
selected groups and spending freed FLOW capacity on MADD selections and path
folds. The first best was candidate 019 at 916 cycles.

All-group overfetch releases 64 FLOW operations. Its candidate 029 combines
selection/path transfers to reduce W to 54,295, but executes in 924 cycles;
the fixed-graph resource-window lower bound is 907. This is a measured example
of scratch and scheduling costs preventing count reductions from becoming
equivalent cycle gains.

Thirteen follow-ups give scalar mix consumers priority after their source
loads and transfer selected remaining MADDs. Candidate 003 reaches **915 cycles**
and is promoted as `results/compact_915`. Its groups are the 16 even-numbered
groups; four depth-four node selections (groups 12–15) change to FLOW.

| Frozen candidate | Cycles | Bundles | W |
| --- | ---: | ---: | ---: |
| `candidate_019` | 916 | 10,809 | 54,609 |
| `candidate_029` | 924 | 10,817 | 54,295 |
| `../overfetch_refine_917/candidate_003` | 915 | 10,808 | 54,577 |

Every listed candidate passed ten seeds with all 20,480 retained hash
checkpoints per seed, complete PC mapping, outputs, preservation of non-output
memory and standalone expansion. A 100000 search-summary entry means no
allocated candidate was retained, not a measured cycle count.

```sh
python3 search_overfetch.py results/memory_vector_followup_917/candidate_007 results/overfetch_917
PYTHONPATH=. python3 results/overfetch_refine_917/run.py
python3 -m unittest test_overfetch -v
python3 export_candidate.py results/compact_915
```
