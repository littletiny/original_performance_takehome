# Retained graph prototypes before scheduling

All configurations are explicit saved graphs choices, not claimed schedules.
`run.py` rebuilds each graph, checks its exact saved resource-bound report and
checks all 20,480 hash checkpoints and non-output memory on seeds 0 and 1.

Starting from `../joint_inventory_915/candidate_047`, `probe_3` through `probe_6`
undo the selected constant LOADs and remove that many depth-5 grandchild
caches. Fewer cache copies save arithmetic, but later node gathers increase
LOAD pressure. The best of these four is bound 903; removing more caches
increases the release/tail LOAD bound to 907, 911 and 915.

The `chain_*` configurations change the lengths of R3/R14 dispatch chains.
`chain_2_2` reaches bound 900 with W=53,820. `setup_*` then vary scalar tree-bias
setup; `setup_1_d3` keeps only the depth-3 bias scalar. Finally `anchors_*`
place the first 2, 4 or 6 additional input-pointer constants on LOAD. The
four-anchor graph reaches bound 899 with W=53,825. This reduces early ALU
work and improves initial vector readiness without shortening the hash.

The configurations are the complete reproduction inputs. The original manual
sequence is recorded above; the rebuild script does not rerun a parameter
search. Later scheduled variants are in `../bound899_915/`.

```sh
python3 results/cache_constant_trade_915/run.py
python3 search_bound899.py
```
