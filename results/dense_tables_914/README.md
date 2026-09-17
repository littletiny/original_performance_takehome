# Remove padding before the tail PC bank

`dense_pc_tables` omits the forced tail singleton-table bank at address 2318.
The first four singleton address banks remain available. Input pointers cease
being artificially grouped into unused tail PC vectors, and scalar I/O values
retain their normal numeric meaning. PC bit pools are disabled for this layout.

The 36-case combined STORE-span inventory has minimum window bound 900 and
minimum W=53,779. Eight selected graphs and a layout-only control each receive
32 trials of 700 iterations. Two-seed semantics precede scheduling.

| Frozen candidate | Cycles | Static bundles |
| --- | ---: | ---: |
| 100, layout only | 914 | 10,167 |
| 024, extended STORE spans | 959 | 11,444 |

Both pass all retained checkpoints and PC mappings on seeds 0–9, final outputs,
preservation of non-output memory and standalone expansion equality. The dense
914 control has W=54,575 and is retained as a smaller-code alternative. It is
not the default because the promoted program has lower arithmetic work and
already meets the bundle cap. This layout makes additional long-handler
configurations fit the cap; it does not by itself improve dynamic cycles.
