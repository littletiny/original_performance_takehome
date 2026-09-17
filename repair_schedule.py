"""Repair a saved schedule inside bounded time windows using CP-SAT.

An infeasible result applies only to the stated windows and fixed graph.
Feasible schedules still need scratch allocation and frozen execution.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np
from ortools.sat.python import cp_model

from optimize import build, Scheduler, RESOURCE_CAPACITY, allocate, counts
from search_compact import static_size


def repair(source, output, cycles, window, seconds, workers, lane_order=False):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source/'config.json').read_text())
    graph = build(config)
    scheduler = Scheduler(graph)
    saved = np.load(source/'best.npz')
    previous = saved['unit_times']
    assert np.array_equal(saved['times'], scheduler.op_times(previous))
    early = np.zeros(len(graph.units), dtype=np.int64)
    for unit in scheduler.order:
        for child, lag in scheduler.children[unit]:
            early[child] = max(early[child], early[unit]+lag)
    model = cp_model.CpModel()
    starts = []
    domains = []
    for u, old in enumerate(previous):
        lo = max(int(early[u]), int(old)-window)
        hi = min(cycles-1-int(scheduler.tail[u]), int(old)+window)
        if lo > hi:
            report = dict(status='EMPTY_DOMAIN', unit=u, lo=lo, hi=hi)
            break
        domains.append((lo, hi))
        var = model.NewIntVar(lo, hi, f't{u}')
        starts.append(var)
        model.AddHint(var, min(hi, max(lo, int(old))))
    else:
        for a, b, lag in zip(scheduler.sources, scheduler.dests, scheduler.lags):
            model.Add(starts[b] >= starts[a]+int(lag))
        order_constraints = 0
        if lane_order:
            packs = defaultdict(list)
            for i, name in enumerate(graph.names):
                if re.search(r'\.lane\d+$', name):
                    packs[name.rsplit('.lane', 1)[0]].append(i)
            for ids in packs.values():
                ids.sort(key=lambda i: (saved['times'][i], i))
                for before, after in zip(ids, ids[1:]):
                    a, b = scheduler.op_units[before], scheduler.op_units[after]
                    model.Add(starts[b]+int(scheduler.op_offsets[after]) >=
                              starts[a]+int(scheduler.op_offsets[before]))
                    order_constraints += 1
        interval_count = 0
        for engine, capacity in enumerate(RESOURCE_CAPACITY):
            intervals, demands = [], []
            for u in range(len(starts)):
                profile = scheduler.usage[u, :, engine]
                offset = 0
                while offset <= scheduler.durations[u]:
                    demand = int(profile[offset])
                    end = offset+1
                    while end <= scheduler.durations[u] and profile[end] == demand:
                        end += 1
                    if demand:
                        intervals.append(model.NewFixedSizeIntervalVar(
                            starts[u]+offset, end-offset, f'p{engine}_{u}_{offset}'))
                        demands.append(demand)
                    offset = end
            model.AddCumulative(intervals, demands, int(capacity))
            interval_count += len(intervals)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = 20260917
        begin = time.monotonic()
        status = solver.Solve(model)
        report = dict(status=solver.StatusName(status), solve_seconds=time.monotonic()-begin,
                      variables=len(starts), intervals=interval_count,
                      lane_order_constraints=order_constraints,
                      branches=solver.NumBranches(), conflicts=solver.NumConflicts())
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            units = np.array([solver.Value(var) for var in starts], dtype=np.int64)
            assert np.all(units[scheduler.dests] >= units[scheduler.sources]+scheduler.lags)
            usage = np.zeros((cycles+33, len(RESOURCE_CAPACITY)), dtype=np.int64)
            for u, start in enumerate(units):
                usage[start:start+33] += scheduler.usage[u]
            assert np.all(usage <= RESOURCE_CAPACITY)
            times = scheduler.op_times(units)
            score = int(times.max())+1
            assert score <= cycles
            bases, allocation = allocate(graph, times)
            report.update(cycles=score, static_bundles=static_size(graph, score), allocation=allocation)
            if bases is not None and report['static_bundles'] <= 12000:
                (output/'config.json').write_text(json.dumps(graph.config, indent=2)+'\n')
                np.savez_compressed(output/'best.npz', unit_times=units, times=times)
                report['allocated'] = True
            else:
                np.savez_compressed(output/'unallocated.npz', unit_times=units, times=times)
                report['allocated'] = False
    report.update(source=str(source), target=cycles, window=window, time_limit_seconds=seconds,
                  lane_order=lane_order,
                  scope='Fixed graph and recorded time windows, plus saved scalar-lane order when enabled; not a global task bound.',
                  counts=counts(graph))
    (output/'repair.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lane-order', action='store_true')
    args = parser.parse_args()
    repair(args.source, args.output, args.cycles, args.window, args.seconds, args.workers, args.lane_order)


if __name__ == '__main__':
    main()
