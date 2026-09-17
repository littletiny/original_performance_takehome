# Early bias constants and memory-materialized shallow tables

Sixteen configurations combine early FLOW bias/address constants, shallow-table
STORE/VLOAD materialization, optional memory-based multiplier 2 and scalar root
mixes. Warm priorities affect search only; they do not alter the graph or ISA.

The minimum W is 54,362 and minimum resource-window bound is 909. No retained
schedule beats the 913 source. `candidate_001` is fully frozen-verified at
**917 cycles / 11,047 bundles**, including ten seeds, all retained checkpoints,
PC mappings, outputs, non-output memory and exact standalone expansion.

```sh
python3 search_prologue_tables.py
python3 export_candidate.py results/prologue_tables_914/candidate_001
```
