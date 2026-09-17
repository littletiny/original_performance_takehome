"""Move independent hash packs into measured ALU/VALU holes, then reschedule."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np

from optimize import build, counts, Scheduler, allocate, CAPACITY, ENGINES, forced_scalar_pack
from search_compact import static_size


def prepare(source, contractions, expansions):
    cfg = json.loads((source/'config.json').read_text())
    graph = build(cfg)
    times = np.load(source/'best.npz')['times']
    total = int(times.max())+1
    usage = np.zeros((total,5), dtype=np.int64)
    writer, readers, incoming, outgoing = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for i, op in enumerate(graph.ops):
        usage[int(times[i]), ENGINES.index(op[0])] += 1
        for value, size in op[3]:
            for j in range(size): writer[int(value)+j] = i
        for value, size in op[2]:
            for j in range(size): readers[int(value)+j].append(i)
    for before, after, lag in graph.control:
        incoming[after].append((before,lag))
        outgoing[before].append((after,lag))
    packs = defaultdict(list)
    pattern = re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    for i, name in enumerate(graph.names):
        stem = name.rsplit('.lane',1)[0]
        if pattern.fullmatch(stem) and graph.ops[i][0] in ('alu','valu'):
            parts=stem.split('.')
            if forced_scalar_pack(cfg,int(parts[0][1:]),int(parts[1][1:]),'.'.join(parts[2:])):
                continue
            packs[stem].append(i)

    def bounds(op_id, lane=None):
        op = graph.ops[op_id]
        reads = [int(v)+j for v,size in op[2] for j in ([lane] if lane is not None else range(size))]
        writes = [int(v)+j for v,size in op[3] for j in ([lane] if lane is not None else range(size))]
        parents = {writer[v] for v in reads if v in writer}
        children = {child for v in writes for child in readers[v]}
        lo = max([0, *[int(times[p])+1 for p in parents],
                  *[int(times[p])+lag for p,lag in incoming[op_id]]])
        hi = min([total-1, *[int(times[c])-1 for c in children],
                  *[int(times[c])-lag for c,lag in outgoing[op_id]]])
        return lo,hi,parents|children|{p for p,_ in incoming[op_id]}|{c for c,_ in outgoing[op_id]}

    selected, blocked = {}, set()
    for scalar, limit in ((False, contractions),(True, expansions)):
        for _ in range(limit):
            best = None
            for name, ids in packs.items():
                if name in selected or blocked.intersection(ids): continue
                engine = graph.ops[ids[0]][0]
                if scalar and engine != 'valu' or not scalar and engine != 'alu': continue
                if scalar:
                    assert len(ids)==1
                    original = int(times[ids[0]])
                    free = 12-usage[:,0].copy()
                    slots = [None]*8
                    rows = [(*bounds(ids[0],j),j) for j in range(8)]
                    neighbors = set()
                    for lo, hi, adjacent, lane in sorted(rows,key=lambda row:row[1]):
                        holes = [t for t in range(lo,hi+1) if free[t]>0]
                        if not holes: break
                        t = min(holes,key=lambda t:(abs(t-original),t))
                        slots[lane] = t
                        free[t] -= 1
                        neighbors.update(adjacent)
                    if None in slots: continue
                    pressure = int(usage[original,1]==6)
                    gain = original-max(slots)
                    distance = sum(abs(t-original) for t in slots)
                else:
                    assert len(ids)==8
                    rows = [bounds(i) for i in ids]
                    lo, hi = max(x[0] for x in rows), min(x[1] for x in rows)
                    holes = [t for t in range(lo,hi+1) if usage[t,1]<6]
                    if not holes: continue
                    t = min(holes)
                    slots = [t]*8
                    neighbors = set().union(*(x[2] for x in rows))
                    pressure = sum(int(usage[int(times[i]),0]==12) for i in ids)
                    gain = max(int(times[i]) for i in ids)-t
                    distance = sum(abs(t-int(times[i])) for i in ids)
                rank = (pressure,gain,-distance,name)
                if best is None or rank > best[0]: best = (rank,name,ids,slots,neighbors)
            if best is None: break
            _, name, ids, slots, neighbors = best
            for i in ids: usage[int(times[i]), ENGINES.index(graph.ops[i][0])] -= 1
            if scalar:
                for t in slots: usage[t,0] += 1
            else:
                usage[slots[0],1] += 1
            selected[name] = scalar,slots
            blocked.update(neighbors)
    cfg['scalar_overrides'] = dict(cfg.get('scalar_overrides') or {})
    cfg['scalar_overrides'].update({name:scalar for name,(scalar,_) in selected.items()})
    changed = build(cfg)
    named = dict(zip(graph.names,map(int,times)))
    placed = []
    for name in changed.names:
        stem = name.rsplit('.lane',1)[0]
        if stem in selected:
            scalar, slots = selected[stem]
            placed.append(slots[int(name.rsplit('.lane',1)[1])] if scalar else slots[0])
        else:
            placed.append(named[name])
    placed = np.array(placed,dtype=np.int64)
    scheduler = Scheduler(changed)
    units = np.array([placed[rows[0][0]]-rows[0][1] for rows in changed.units],dtype=np.int64)
    assert np.array_equal(scheduler.op_times(units),placed)
    assert np.all(units[scheduler.dests]>=units[scheduler.sources]+scheduler.lags)
    measured = np.zeros_like(usage)
    for i,op in enumerate(changed.ops): measured[int(placed[i]),ENGINES.index(op[0])] += 1
    assert np.array_equal(usage,measured) and np.all(usage<=CAPACITY)
    return changed,scheduler,units,selected


def run(job):
    source, output, index, contract, expand = job
    graph,scheduler,incumbent,selected = prepare(source,contract,expand)
    directory = output/f'candidate_{index:03}'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'config.json').write_text(json.dumps(graph.config,indent=2)+'\n')
    (directory/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    best_score = 100000
    for trial in range(20):
        if trial:
            score,best,last = scheduler.search(incumbent,200,index*100+trial,(.1,.5,1,2)[trial%4])
        else:
            best=incumbent
            score=int((best+scheduler.durations).max())+1
        times=scheduler.op_times(best)
        bases,audit=allocate(graph,times)
        if bases is not None and score<best_score and static_size(graph,score)<=12000:
            best_score=score
            incumbent=best
            np.savez_compressed(directory/'best.npz',unit_times=best,times=times)
            row=dict(candidate=index,cycles=score,contractions=contract,expansions=expand,
                     changed_packs=len(selected),static_bundles=static_size(graph,score),
                     **audit,**counts(graph))
            (directory/'search.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    return index,best_score


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--moves',type=json.loads,
                        help='JSON list of [ALU-to-VALU packs, VALU-to-ALU packs]')
    args=parser.parse_args()
    options=args.moves or [(0,8),(0,16),(0,32),(8,0),(16,0),(32,0),(8,16),(16,24),(32,40),(48,56)]
    assert all(len(row)==2 and all(isinstance(x,int) and x>=0 for x in row) for row in options)
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,[(args.source,args.output,i,a,b) for i,(a,b) in enumerate(options)]))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
