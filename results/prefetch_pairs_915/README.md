# Early depth-5 sibling addresses and adjacent-pair prefetch

This experiment changes when depth-5 addresses and nodes become ready. The
submitted production kernel remains at **915 cycles / 10,808 bundles**.

With the existing compact mirrored heap, the folded depth-5 address is
`32 + 4*q3 + 2*b3 + b4`. After round 3, a FLOW selection and MADD can form
the prefix `32 + 4*q3 + 2*b3`. Round 4 then adds `b4`. Relative to the existing
folded version, this moves the address work earlier at +8 W/group and frees
two FLOW selections/group. `early_prefix_groups` selects this behavior.

`prefetch_pair_groups` goes further. Eight real VLOAD instructions read both
possible children, one per input lane, into a 22-word scratch span at offsets
0,2,...,14. Ordered later writes overwrite preceding padding, leaving
`[L0,R0,...,L7,R7]` in words 0..15. Two real eight-lane FLOW selections consume
the even output lanes. Eight scalar XORs produce the next hash input. Every
real eight-word write is declared and allocated, including unused padding.
The native allocator now chooses a sufficient word stride for the virtual span;
the machine itself still has eight lanes and 1,536 scratch words.

Compared with the old folded path, packed prefetch has the same LOAD and FLOW
instruction counts and adds 8 W/group. It may shorten the load dependency,
but increases scratch lifetime and writes, so lower latency is not guaranteed.
No ISA, capacity, cycle-counting or test changes are involved.

`search_prefetch_pairs.py` screens 16 local and combined configurations, each
with two-seed reference semantics and 24 native trials of 500 iterations.
All configurations enforce the 12,000-bundle cap. No result beats production.
Seven configurations retain no allocated schedule; `100000` in the summary
is a sentinel, not a measured score.

| Retained candidate | Change | Verified cycles | Static bundles |
| --- | --- | ---: | ---: |
| 000 | Early prefixes for two groups | 915 | 10,808 |
| 006 | Packed prefetch for eight groups | 917 | 10,810 |
| 014 | Packed prefetch with the combined reduction graph | 968 | 11,901 |

All three rows passed seeds 0–9 on the frozen machine: 20,480 retained hash
checkpoints per seed, all executed PCs, outputs, all non-output memory and
standalone expansion equality. Other scores in `summary.json` are allocated
screens, not independently frozen-verified exports.

`test_prefetch_pairs.py` checks real overlapping VLOAD writes, both choices,
untouched guard memory and both fixed and allocated scratch. A full-graph
negative control delays one load, removes its required ordering edge and
detects an incorrect node at round 5's first hash checkpoint. The 21 relevant
regression tests pass; see `regression_tests.log`.

```sh
python3 test_prefetch_pairs.py
python3 search_prefetch_pairs.py
python3 export_candidate.py results/prefetch_pairs_915/candidate_000
python3 export_candidate.py results/prefetch_pairs_915/candidate_006
python3 export_candidate.py results/prefetch_pairs_915/candidate_014
```
