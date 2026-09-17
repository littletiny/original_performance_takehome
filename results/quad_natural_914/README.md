# Independent table placement and execution order

`natural_pc_order` keeps PC offset vectors in ordinary lane order while
interleaved row producers visit lanes 0,2,4,6,1,3,5,7. Per-region `table_lanes`
maps each executed phase to its physical table segment. The lowerer and
standalone exporter both use that mapping. Unrelated handlers can retain their
original order. `quad_row_order` independently controls temporary-row reuse.

The 16-case screen varies two/three-group selections, PC layout and sorted
versus measured producer order. It does not beat the 916-cycle quartet control
or production 914. `candidate_015` is fully frozen-verified at **917 cycles /
11,314 bundles**, W=54,431, with three-group selection and natural PC layout.
Seeds 0–9 pass every retained hash checkpoint, PC mapping, output and non-output
memory check; standalone expansion is identical.

The real-machine PC microtest also covers a permuted execution order with
natural physical table positions, non-unit case stride and a wrapped MADD bias.

```sh
python3 results/quad_natural_914/run.py
python3 export_candidate.py results/quad_natural_914/candidate_015
```
