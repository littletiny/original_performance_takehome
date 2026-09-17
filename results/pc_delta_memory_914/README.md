# Time the PC addend's memory broadcast

Starting with the verified PC-offset candidate, `run.py` moves its addend vector
2462 through the reusable STORE/VLOAD row. It tests seven row positions and
two scalar/vector balance policies: 14 configurations, each with two reference
seeds and 32 trials of 750 iterations.

Candidates 000, 003 and 007 retain 914 in the screen. Candidate 000 is fully
frozen-verified at **914 cycles / 10,807 bundles**, W=54,484. Seeds 0–9 pass
every retained checkpoint, PC mapping, output, non-output memory and standalone
expansion check. Late placement of this early PC dependency performs worse.

One subsequent ALU-to-VALU hash-pack transfer lowers the resource-window bound
to 910 while retaining 914. That result is promoted in `../compact_914_pc/`.
