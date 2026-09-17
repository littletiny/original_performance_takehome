"""Exchange grandchild caches for retained raw tree blocks."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search


def main():
    source = Path('results/compact_918')
    output = Path(__file__).resolve().parent
    base = json.loads((source/'config.json').read_text())
    original = build(base)
    forms = dict(base['scalar_overrides'])
    pattern = re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    for name, op in zip(original.names, original.ops):
        stem = name.rsplit('.lane', 1)[0]
        if pattern.fullmatch(stem):
            forms[stem] = op[0] == 'alu'
    base = dict(base, scalar_overrides=forms)
    unchanged = build(base)
    assert unchanged.ops == original.ops and unchanged.names == original.names
    jobs, screen = [], []
    grand = base['prefetch5_groups']
    for keep in ([5, 6], [5, 6, 7], [4, 5, 6, 7]):
        for number in (0, 2, 4, 6):
            selected = [grand[i*len(grand)//number] for i in range(number)]
            for policy in (0, 1):
                config = dict(base, prefetch5_groups=selected, heap_keep_levels=keep,
                              prefetch_madd_groups=[], precise_restore=True,
                              lane_allocation_trials=16)
                config = balance(config, source, policy)
                graph = build(config)
                verify_semantics(graph, (0,))
                index = len(jobs)
                screen.append(dict(candidate=index, keep=keep, grand=selected,
                                   policy=policy, **counts(graph)))
                jobs.append((index, config, source, output, 12, 400))
    (output/'screen.json').write_text(json.dumps(screen, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(search, jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows, key=lambda x:x[1]), indent=2)+'\n')
    print('RESULT', sorted(rows, key=lambda x:x[1]), flush=True)


if __name__ == '__main__':
    main()
