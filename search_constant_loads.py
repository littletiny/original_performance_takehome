"""Move selected scalar constant synthesis to spare LOAD slots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import re

import numpy as np

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def configurations(source):
    base = json.loads((source/'config.json').read_text())
    graph = build(base)
    scheduler = Scheduler(graph)
    saved = np.load(source/'best.npz')['times']
    pattern = re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    forms = dict(base.get('scalar_overrides') or {})
    for name, op in zip(graph.names, graph.ops):
        stem = name.rsplit('.lane', 1)[0]
        if pattern.fullmatch(stem):
            forms[stem] = op[0] == 'alu'
    base = dict(base, scalar_overrides=forms)
    retained = build(base)
    assert graph.names == retained.names and graph.ops == retained.ops
    constants = [(int(name.split('.')[1]), i) for i, (name, op) in enumerate(zip(graph.names, graph.ops))
                 if re.fullmatch(r'constant\.\d+', name) and op[0] == 'alu']
    readers = {}
    for i, op in enumerate(graph.ops):
        for value, size in op[2]:
            for j in range(size):
                readers.setdefault(int(value)+j, []).append(i)
    orders = {}
    orders['early'] = sorted(constants, key=lambda p: (saved[p[1]], p[0]))
    orders['late'] = list(reversed(orders['early']))
    orders['tail'] = sorted(constants, key=lambda p: (-scheduler.tail[scheduler.op_units[p[1]]], p[0]))
    orders['fanout'] = sorted(constants, key=lambda p: (-len(readers.get(int(graph.ops[p[1]][1][1]), [])), p[0]))
    rng = random.Random(20260917)
    orders['random'] = list(constants)
    rng.shuffle(orders['random'])
    seen = set()
    for name, order in orders.items():
        for number in (16, 32, 48, 64, 80):
            chosen = tuple(sorted(value for value, _ in order[:number]))
            if chosen in seen:
                continue
            seen.add(chosen)
            for policy in (0, 1):
                config = balance(dict(base, force_load_scalars=chosen), source, policy)
                yield dict(selection=name, number=number, policy=policy), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs, screen = [], []
    for description, config in configurations(args.source):
        graph = build(config)
        verify_semantics(graph, (0,))
        screen.append(dict(candidate=len(configs), **description, **counts(graph)))
        configs.append(config)
    (args.output/'screen.json').write_text(json.dumps(screen, indent=2)+'\n')
    jobs = [(i, cfg, args.source, args.output, args.trials, args.iterations)
            for i, cfg in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(search, jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows, key=lambda x:x[1]), indent=2)+'\n')
    print('RESULT', sorted(rows, key=lambda x:x[1]), flush=True)


if __name__ == '__main__':
    main()
