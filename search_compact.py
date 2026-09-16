"""Offline search for schedules with at most 10,000 static VLIW bundles."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random

import numpy as np

from optimize import build, counts, Scheduler, allocate


def warm_keys(source, graph):
    old = build(json.loads((source / 'config.json').read_text()))
    times = np.load(source / 'best.npz')['times']
    named = dict(zip(old.names, map(int, times)))
    for name, t in list(named.items()):
        if '.lane' in name:
            stem = name.rsplit('.lane', 1)[0]
            named[stem] = min(named.get(stem, t), t)
    return np.array([named.get(graph.names[rows[0][0]],
                              named.get(graph.names[rows[0][0]].rsplit('.lane', 1)[0], 0))
                     for rows in graph.units], dtype=np.float64)


def static_size(graph, cycles):
    holes = 8 * len(graph.regions) if graph.config['compact_main'] else 0
    return cycles + graph.total_table_words - holes


def search(job):
    index, config, source, output, trials, iterations = job
    graph = build(config)
    scheduler = Scheduler(graph)
    rng = random.Random(733 + index)
    incumbent = warm_keys(source, graph)
    best_score = 100000
    directory = output / f'candidate_{index:03}'
    for trial in range(trials):
        if trial and trial % 5 == 0:
            keys = scheduler.consumer_setup(scheduler.tags[:, 0] * rng.choice((1, 2, 4)) + scheduler.tags[:, 1])
        else:
            keys = incumbent
        score, best, last = scheduler.search(keys, iterations, rng.randrange(1 << 32), rng.choice((.1, .5, 1, 2, 4)))
        if trial == 0 or score < min(incumbent.max() + 1, best_score):
            incumbent = best
        elif rng.random() < .2:
            incumbent = last
        if score >= best_score or static_size(graph, score) > 10000:
            continue
        times = scheduler.op_times(best)
        bases, audit = allocate(graph, times)
        if bases is None:
            continue
        best_score = score
        incumbent = best
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(graph.config, indent=2) + '\n')
        np.savez_compressed(directory / 'best.npz', unit_times=best, times=times)
        report = dict(candidate=index, trial=trial, cycles=score,
                      static_bundles=static_size(graph, score), **audit, **counts(graph))
        (directory / 'search.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    return index, best_score


def configurations(base):
    for jump5, pairs in ((0, 14), (2, 13), (4, 11)):
        for leaf in (.125, .25, .375, .5):
            for direct in ((), (64, 128, 1024)):
                for position in ('early', 'spread'):
                    options = [(r, k) for r in (3, 14) for k in range(0, 28 if r == 14 else 32, 2)]
                    if position == 'early':
                        options.sort(key=lambda row: (row[1], row[0]))
                    else:
                        options = options[::2] + options[1::2]
                    overrides = [(r, k, 2) for r, k in options[:pairs]]
                    config = dict(base, width3=1, width5=1, jump5=jump5,
                                  path2_flow=False, fuse_tail_pc=False,
                                  pair_early_end=False, last_pair3=False,
                                  fold_path4=False, leaf_madd=leaf,
                                  dispatch_widths=overrides, compact_main=True,
                                  load_vectors=direct)
                    e = counts(build(config))['engines']
                    config['scalar'] = round(config['scalar'] + (8 * e['valu'] - 4 * e['alu']) / 160000, 5)
                    yield config


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('results/portfolio_1'))
    p.add_argument('--output', type=Path, default=Path('results/compact_search'))
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--iterations', type=int, default=200)
    args = p.parse_args()
    base = json.loads((args.source / 'config.json').read_text())
    configs = list(configurations(base))
    jobs = [(i, c, args.source, args.output, args.trials, args.iterations) for i, c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(search, jobs))
    print(json.dumps(sorted(results, key=lambda row: row[1])), flush=True)
