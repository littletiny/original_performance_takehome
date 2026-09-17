# Earlier depth-5 loads on the combined graphs

Twelve graphs add zero/four/eight/sixteen early paired-node prefetch groups to
three retained joint configurations. Every graph passes reference seeds 0/1;
three pass the screen's window-bound and static-size gates and receive 24
trials of 600 iterations.

Earlier loads reduce the lowest-work graph's window bound from 923 to 921,
while increasing arithmetic work. They do not repair the resource deficit.
The best bound in this inventory remains 901. Two allocated screens are 1,049
and 1,050 cycles; the third retains no allocation.

Candidate 002 is frozen-verified at **1,049 cycles / 11,998 bundles**, with all
retained checkpoints, PC mappings, outputs and non-output memory checked on
seeds 0–9, plus exact standalone expansion. It exercises the combined early
child rows, 22-word prefetch spans, even/odd PC banks and extended handlers.
It is a correctness/control artifact and is not promoted.
