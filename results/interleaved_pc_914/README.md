# Reuse scalar buffer addresses as dispatch PC offsets

Eight singleton tables share an interleaved physical bank. For bank b and lane
j, the target is `64*q + 2311 + 8*b + j`. The offset words 2311–2374 already serve
the child-store buffers and input addresses. Grouping those scalar words into
vectors avoids four former vector additions without synchronizing the input
groups' hash execution. The tables remain distinct and contain runtime lookups.

The lowerer/exporter support independent lane and case strides. Every table
cell is checked for range and collision. A real-machine test exhausts all cases
of two interleaved tables in normal and permuted execution orders, including
output and non-output memory checks; a colliding bank is rejected.

Twenty-four configurations vary bank size, selected groups and node-selection
MADDs. Their minimum W is 54,344 and minimum resource-window bound is 908; these
metrics alone are not executed scores. `candidate_001` is fully frozen-verified
at **914 cycles / 11,056 bundles**, W=54,427, bound 909. All ten seeds pass every
retained checkpoint and memory/PC check, and standalone expansion is identical.

Wider banks introduce additional scalar constants;
none beats 914 in this screen. `100000` is a search failure sentinel. Subsequent
prefix work produces the promoted 913 checkpoint in `../compact_913_packed/`.

```sh
python3 search_interleaved_pc.py
python3 export_candidate.py results/interleaved_pc_914/candidate_001
```
