# Local STORE-span controls

`run.py` makes 12 small changes around production 915: two/three-cycle bodies
for one/three/six grandchild caches, optionally with two late paired-child rows.
It retains depth-6 originals to make the extended input buffer disjoint from
tree backups. Each graph passes two-seed semantics and receives 24 native
trials of 500 iterations.

Candidate 002 is frozen-verified at **920 cycles / 11,173 bundles**, W=54,556.
It uses span two for three early singleton regions. Seeds 0–9 pass all retained
hash checkpoints, PC mapping, outputs, non-output memory and standalone
expansion equality. It does not improve production.

The other 11 screens retain no allocated schedule; `100000` in `summary.json`
is an absence marker, not a measured cycle count. Complete parameter choices
can be regenerated with `python3 results/store_spans_local_915/run.py`.
