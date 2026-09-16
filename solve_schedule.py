"""CP-SAT schedule refinement for the offline SSA graph, with frozen checks."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from ortools.sat.python import cp_model

from optimize import build, Scheduler, allocate, counts, RESOURCE_CAPACITY


def warm(source, config):
    old = build(json.loads((source / 'config.json').read_text()))
    old_times = np.load(source / 'best.npz')['times']
    named = dict(zip(old.names, map(int, old_times)))
    for name, t in list(named.items()):
        if '.lane' in name:
            stem = name.rsplit('.lane', 1)[0]
            named[stem] = min(named.get(stem, t), t)
    graph = build(config)
    scheduler = Scheduler(graph)
    keys = []
    for rows in graph.units:
        name = graph.names[rows[0][0]]
        keys.append(named.get(name, named.get(name.rsplit('.lane', 1)[0], 0)))
    score, best, _ = scheduler.search(keys, 1000, 617, 1.0)
    return graph, scheduler, score, best


def solve(source, output, config, seconds, workers, cutoff=None):
    output.mkdir(parents=True, exist_ok=True)
    graph, scheduler, score, seed = warm(source, config)
    (output/'config.json').write_text(json.dumps(graph.config, indent=2)+'\n')
    np.savez_compressed(output/'best_ir.npz', unit_times=seed, times=scheduler.op_times(seed))
    bases, audit = allocate(graph, scheduler.op_times(seed))
    if bases:
        np.savez_compressed(output/'best.npz', unit_times=seed, times=scheduler.op_times(seed))
    print(json.dumps(dict(initial_cycles=score, allocation=audit, **counts(graph))), flush=True)
    model = cp_model.CpModel()
    horizon = cutoff or score
    n = len(graph.units)
    starts = [model.NewIntVar(0, horizon-1-int(scheduler.durations[u]), f't{u}') for u in range(n)]
    end = model.NewIntVar(0, horizon, 'cycles')
    for u in range(n):
        model.Add(starts[u] + int(scheduler.durations[u]) + 1 <= end)
        if seed[u]+scheduler.durations[u] < horizon:
            model.AddHint(starts[u], int(seed[u]))
    if score <= horizon:
        model.AddHint(end, score)
    for a, b, lag in zip(scheduler.sources, scheduler.dests, scheduler.lags):
        model.Add(starts[int(b)] >= starts[int(a)] + int(lag))
    for engine, capacity in enumerate(map(int,RESOURCE_CAPACITY)):
        intervals, demands = [], []
        for u in range(n):
            row = scheduler.usage[u, :, engine]
            p = 0
            while p <= scheduler.durations[u]:
                if not row[p]:
                    p += 1
                    continue
                q = p+1
                while q <= scheduler.durations[u] and row[q] == row[p]:
                    q += 1
                intervals.append(model.NewIntervalVar(starts[u]+p, q-p, starts[u]+q, f'e{engine}.{u}.{p}'))
                demands.append(int(row[p]))
                p = q
        model.AddCumulative(intervals, demands, capacity)
    model.Minimize(end)

    class Solutions(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.best = 100000
            self.started = time.monotonic()

        def on_solution_callback(self):
            cycles = self.Value(end)
            if cycles >= self.best:
                return
            self.best = cycles
            times = np.array([self.Value(v) for v in starts], dtype=np.int64)
            op_times = scheduler.op_times(times)
            np.savez_compressed(output/'best_ir.npz', unit_times=times, times=op_times)
            bases, audit = allocate(graph, op_times)
            row = dict(cycles=cycles, seconds=time.monotonic()-self.started, **audit)
            if bases:
                np.savez_compressed(output/'best.npz', unit_times=times, times=op_times)
                (output/'result.json').write_text(json.dumps(row, indent=2)+'\n')
            print(json.dumps(row), flush=True)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 233
    solver.parameters.log_search_progress = False
    status = solver.Solve(model, Solutions())
    report = dict(status=solver.StatusName(status), bound=solver.BestObjectiveBound(), seconds=solver.WallTime())
    (output/'solver.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('results/portfolio_1'))
    p.add_argument('--output', type=Path, default=Path('results/cpsat'))
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--config', default='{"scalar":0.286}')
    p.add_argument('--cutoff', type=int)
    args = p.parse_args()
    config = json.loads((args.source/'config.json').read_text())
    config.update(json.loads(args.config))
    solve(args.source, args.output, config, args.seconds, args.workers, args.cutoff)
