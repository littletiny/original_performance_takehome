# Producer-first dispatch-chain control

Starting from `../quad_joint_914/candidate_123`, pair each round-3 quartet
producer, in group order, with the next non-quartet round-3 region. Keep the
round-14 chains unchanged. The saved configuration defines the complete graph.

Both semantic seeds pass every retained hash checkpoint and preserve non-output
memory. W stays 53,924. The LOAD window changes to release 76, tail 65 and work
1,524, yielding bound **903**, worse than the source graph's 902. No allocated
or frozen schedule is claimed for this control.

```sh
python3 resource_bounds.py results/quad_producers_first_914 --output results/quad_producers_first_914/bounds_rebuilt.json
```
