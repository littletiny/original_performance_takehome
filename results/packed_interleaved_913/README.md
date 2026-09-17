# Fill table holes without additional instructions

The last wide table moves from PC 9991 to 1285. Its preceding offset vector is
`9479 + 64*j`; taking modulo the already-live hash multiplier 4097 gives
`1285 + 64*j` for every lane. This replaces the former vector addition with one
vector modulo operation, using the unchanged ISA.

Three singleton tables formerly starting at 1230, 1294 and 1358 move to 1932,
1996 and 2060. The first uses the already-live constant vector 766; the next two
retain their normal 64-word increments. The builder proves lane arithmetic and
table disjointness before lowering, whose collision checks cover every case.

Exactly two operations change: `r14.g15.dispatch.offsets` and
`r14.g24.dispatch.offsets`. All operation times and engine counts remain exact.
The result is **913 cycles / 10,537 bundles**, 512 fewer bundles than its source,
with unchanged W=54,406, live peak 1,322 and 231,523 static slot operations.
Ten frozen seeds pass every retained checkpoint, PC and memory check, and the
standalone decoder matches. It is promoted in `../compact_913_packed/`.

```sh
python3 results/packed_interleaved_913/run.py
python3 export_candidate.py results/packed_interleaved_913
```
