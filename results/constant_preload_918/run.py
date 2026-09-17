"""Prioritize constant loads into the measured early LOAD holes."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from optimize import build, counts, verify_semantics
from search_compact import search
from search_constant_loads import configurations


def main():
    source = Path('results/compact_918')
    output = Path(__file__).resolve().parent
    jobs, screen = [], []
    for description, config in configurations(source):
        if description['selection'] not in ('late', 'tail', 'fanout'):
            continue
        if description['number'] not in (48, 64) or description['policy']:
            continue
        for priority in (0, 50):
            cfg = dict(config, constant_load_priority=priority)
            graph = build(cfg)
            verify_semantics(graph, (0,))
            index = len(jobs)
            screen.append(dict(candidate=index, priority=priority, **description, **counts(graph)))
            jobs.append((index, cfg, source, output, 8, 300))
    (output/'screen.json').write_text(json.dumps(screen, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(search, jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows, key=lambda x:x[1]), indent=2)+'\n')
    print('RESULT', sorted(rows, key=lambda x:x[1]), flush=True)


if __name__ == '__main__':
    main()
