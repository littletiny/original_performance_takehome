# FLOW transfers around the first 914-cycle schedule

`run.py` screens 24 combinations: retained/removed early address prefixes,
12/8/4/0 MADD node selections and either no path transfer or eight shallow path
updates in round 1 or 12. All pass two reference seeds before 24 native trials
of 600 iterations; the real static bundle cap and scratch limit are enforced.

Candidate 014 retains **914 cycles / 10,807 bundles** and lowers W from 54,592
to **54,536**. It keeps the two early prefixes and all 12 MADD node selections;
eight round-12 path updates move from MADD to FLOW. The fixed-graph resource
bound falls from 912 to 911. No candidate beats 914.

The selected candidate passes full frozen execution on seeds 0–9, including
every retained hash checkpoint, PC mapping, output and all non-output memory,
plus standalone expansion equality. It is promoted to `../compact_914/`, where
the actual submitted builder is additionally checked on seeds 901 and 12345.
Other results in `summary.json` are allocated screens.
