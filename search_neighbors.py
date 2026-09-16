"""Refine a compact checkpoint with small, individually measured changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from optimize import build
from search_compact import search


def configurations(source):
    base = json.loads((source/'config.json').read_text())
    yield base
    for k in base['prefetch5_groups']:
        yield dict(base, prefetch5_groups=[x for x in base['prefetch5_groups'] if x != k])
    for pair in base['prefetch_madd_groups']:
        yield dict(base, prefetch_madd_groups=[x for x in base['prefetch_madd_groups'] if x != pair])
    for keep in ([4,5], [5,6], [4,5,6], [5,7]):
        if keep != base['heap_keep_levels']:
            yield dict(base, heap_keep_levels=keep)
    # A short gap and an already-ready target make the extra entry jump
    # a useful candidate for removal. Dependencies still decide feasibility.
    if not base['merge_chains']:
        graph = build(base)
        times = np.load(source/'best.npz')['times']
        writer = {int(v)+j:i for i, op in enumerate(graph.ops)
                  for v, size in op[3] for j in range(size)}
        regions = sorted(graph.regions, key=lambda r:int(times[r['start']]))
        pairs = []
        for a, b in zip(regions, regions[1:]):
            gap = int(times[b['start']]-times[a['start']])
            target = graph.ops[b['start']][2][0][0]
            slack = int(times[b['start']]-times[writer[int(target)]])-1
            if gap <= 16:
                pair = [[r['round'],r['groups'][0]] for r in (a,b)]
                pairs.append((gap-slack, gap, pair))
        for _, _, pair in sorted(pairs)[:8]:
            yield dict(base, merge_chains=[pair])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    unique = {}
    for config in configurations(args.source):
        unique.setdefault(json.dumps(config,sort_keys=True),config)
    jobs = [(i, config, args.source, args.output, args.trials, args.iterations)
            for i, config in enumerate(unique.values())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)
