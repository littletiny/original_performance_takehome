# Combined work-reduction inventory

`inventory_joint.py` builds 240 combinations of all-group deep overfetch,
depth-4 path folding, FLOW node selection, late paired-child rows, memory
broadcasts, scalar uniform hash operands and scalar roots. It tunes shallow
path updates against FLOW targets 867, 875 and 877 before exact ALU/VALU balance.

No combination passes the strict resource-window bound <=899. The best retained
bound is **902**, with W=53,893 and 11,832 static bundles at a hypothetical
899-cycle schedule. All 12 retained configurations pass reference semantics
on seeds 0 and 1, including all 20,480 retained hash checkpoints and all
non-output memory. No allocated or frozen cycle score is claimed here.

The lower-bound report captures ALU, VALU, LOAD, STORE, FLOW and weighted
arithmetic release/tail windows. Candidate 047 illustrates the conflict:
arithmetic needs at least 901 cycles, LOAD at least 902, and FLOW at least 900.
Moving more early vector setup onto memory lowers arithmetic but raises the
FLOW startup window to 916–918 in other combinations.

`summary.json` contains every screen result. Reproduce with
`python3 inventory_joint.py`. Further cache/constant/chain prototypes are in
`../cache_constant_trade_915/` and the allocated control in `../bound899_915/`.
