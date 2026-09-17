# Scalar operations share uniform operands

The optional `share_uniform_operands` rewrite replaces ALU reads of repeated
vector lanes with the underlying runtime scalar, or a canonical numeric
constant. A broadcast can disappear when all its consumers use scalar ALU.
`scalar_constant_labels` and `scalar_setup` keep the requested packs scalar;
resource rebalancing now recognizes the full multicomponent hash-stage label.

Sixteen combinations were screened for 20 trials of 500 iterations. The
best retained allocated score is 915; none improve production. The saved
scores here are screens, not independently frozen-verified exports. Forcing
a whole stage may remove one broadcast but can worsen the schedule despite
global ALU/VALU rebalancing.

The regression verifies that all 512*8 `h2.a` scalar operations share one
constant word, its vector broadcast disappears and both reference seeds pass.
Some explicitly requested memory-vector fills become unused after rewrites;
this screen keeps those fills, while the subsequent joint inventory filters
names against a build with memory broadcasts disabled.

```sh
python3 test_uniform_operands.py
python3 search_uniform_operands.py
```
