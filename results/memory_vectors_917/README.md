# Broadcasting through STORE and VLOAD

`memory_vectors` replaces selected broadcasts or uniform constant-vector
expressions with eight scalar STOREs followed by one VLOAD. It uses bounded
rows in the unused index area starting at 2054. All fills precede their read;
a subsequent fill may share the preceding read's cycle. Every used row is
cleared with the initial zero vector after its last read.

Only the research interpreter needed a missing ordinary scalar-STORE case;
the frozen ISA, port capacities and cycle accounting are unchanged. A delayed
runtime-root read passes with the reuse edges and fails a hash checkpoint when
those edges are removed (`test_memory_vectors.py`).

## Results

All 30 broad configurations and 18 later-use configurations passed two-seed
reference semantics before scheduling. Moving early hash constants delays
startup. The broad search's best candidate is 929, while moving every uniform
vector saves 267 W but executes in 994 cycles. Its fixed graph has a stronger
VALU release/tail lower bound of 924, despite a count-only bound of 907.

The later-use search keeps critical startup broadcasts in registers and can
advance fills using `memory_vector_lead`. Candidate 007 transfers ten vectors,
preserving 917 cycles and 10,810 bundles while reducing W from 54,652 to
54,585 (67 fewer). It adds ten LOAD and 81 STORE operations, leaves FLOW
unchanged, and has a fixed-graph lower bound of 912.

All three listed candidates passed seeds 0–9, 20,480 hash checkpoints per seed,
complete PC mapping, final outputs, non-output memory, and standalone expansion:

| Candidate | Cycles | Bundles | W |
| --- | ---: | ---: | ---: |
| `candidate_007` | 929 | 10,822 | 54,595 |
| `candidate_018` | 994 | 10,887 | 54,385 |
| `../memory_vector_followup_917/candidate_007` | 917 | 10,810 | 54,585 |

Six measured-hole rebalances and 15 PC-fold combinations retained 917 as the
best result. This lower-work candidate became the source for the overfetch
experiments, which subsequently produced the verified 915 checkpoint.

```sh
python3 search_memory_vectors.py results/compact_917 results/memory_vectors_917
PYTHONPATH=. python3 results/memory_vector_followup_917/run.py
python3 -m unittest test_memory_vectors -v
python3 export_candidate.py results/memory_vector_followup_917/candidate_007
```

The adjacent follow-up directory contains resource bounds and a full-program
identity check of the then-current 917 production kernel. The original root
utilization artifacts, including their pre-existing timestamp edit, were not
overwritten.
