# Early child-pair rows with contiguous branch bits

The paired handler order `0,2,4,6,1,3,5,7` places even logical lanes in the first
pair block and odd lanes in the second. The condition views are the original
contiguous bit vector and that vector plus one word. Its ninth ignored word
remains inside the declared span. Early path arithmetic can consume the normal
bit vector, so no bit repacking or duplicate masks are needed.

`early_pair_groups` enables this layout at round 3. It requires the even/odd
order, span-one two-stream handlers, ordinary depth-4 children and no grandchild
cache in those streams. `pair_even_odd_order` changes paired handlers; the
separate `all_even_odd_order` option also permutes singleton PC/address banks.
Every case still executes eight lane steps and has the same static case count.

Each selected pair removes 16 child-copy ALU operations, adds two VLOADs and
two FLOW node selections when replacing FLOW choices, before shared setup and
MADD-choice differences. All real VSTORE writes and ignored VSELECT outputs
are accounted for. Early and late rows share an explicitly ordered, cleared
buffer separate from memory broadcasts and the shallow cache.

The real-machine fixture covers original/permuted order, separate/merged
handlers, both child choices, fixed/native allocation and guard memory. A
negative control delays an old odd-lane block read, removes its reuse edge
and fails at round 4's first hash checkpoint.
An additional negative control covers the transition from the last early row
to a custom late-row order; its first writer must retain that old read boundary.

Sixteen configurations receive two-seed semantics and 32 trials of 750
iterations. No result beats 914. Frozen-verified controls are:

| Candidate | Change | Cycles | W |
| --- | --- | ---: | ---: |
| 001 | Two late pairs, new order | 914 | 54,522 |
| 011 | Two early and two late pairs on the broad overfetch source | 917 | 54,264 |

Both pass seeds 0–9, all 20,480 retained checkpoints per seed, PC mapping,
outputs, non-output memory and standalone expansion equality. No early-pair
row or permutation is enabled in the promoted `compact_914_pc`.
