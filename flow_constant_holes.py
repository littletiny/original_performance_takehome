"""Replace scalar-constant ALU work in measured free FLOW or LOAD slots.

Every other surviving operation keeps its exact saved time. All dependencies,
engine capacities, scratch allocation and bootstrap placement are rechecked.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from optimize import build,Scheduler,RESOURCE_CAPACITY,allocate,bootstrap_cycle,counts,verify_semantics
from search_compact import search,static_size
from search_overfetch_levels import applicable_setup


def prepare(source,output,index,budget,engine='flow'):
    base=json.loads((source/'config.json').read_text())
    assert engine in ('flow','load')
    assert engine!='flow' or base.get('pc_prologue')
    graph=build(base);scheduler=Scheduler(graph);times=np.load(source/'best.npz')['times']
    named=dict(zip(graph.names,map(int,times)));cycles=int(times.max())+1
    readers=defaultdict(list);incoming=defaultdict(list);outgoing=defaultdict(list)
    capacity=1 if engine=='flow' else 2
    occupied=np.bincount([int(t) for t,op in zip(times,graph.ops) if op[0]==engine],minlength=cycles)
    if engine=='flow':occupied[bootstrap_cycle(graph,times)]+=1
    for i,op in enumerate(graph.ops):
        for value,length in op[2]:
            for j in range(length):readers[int(value)+j].append(i)
    for a,b,lag in graph.control:
        incoming[b].append((a,lag));outgoing[a].append((b,lag))
    candidates={}
    for i,(name,op) in enumerate(zip(graph.names,graph.ops)):
        if not name.startswith('constant.') or op[0]!='alu':continue
        if len(graph.units[int(scheduler.op_units[i])])!=1:continue
        lo=max([0,*[int(times[a])+lag for a,lag in incoming[i]]])
        hi=min([cycles-1,*[int(times[c])-1 for value,length in op[3]
                           for j in range(length) for c in readers[int(value)+j]],
                *[int(times[b])-lag for b,lag in outgoing[i]]])
        candidates[name]=(int(name.split('.')[1]),lo,hi)
    selected={}
    for t in range(cycles):
        if len(selected)>=budget:break
        for _ in range(capacity-int(occupied[t])):
            if len(selected)>=budget:break
            eligible=[name for name,(_,lo,hi) in candidates.items() if lo<=t<=hi and name not in selected]
            if eligible:
                name=min(eligible,key=lambda name:(candidates[name][2],abs(named[name]-t),name))
                selected[name]=t
    original_flow=set(base.get('flow_constants') or [])
    selected_values={candidates[name][0] for name in selected}
    # FLOW add_imm is normally retained as a FLOW root. Drop conversions that
    # became unused after a dependent constant was also rematerialized.
    for _ in range(len(selected)+1):
        raw=(dict(base,flow_constants=sorted(original_flow|selected_values)) if engine=='flow' else
             dict(base,force_load_scalars=sorted(set(base.get('force_load_scalars') or [])|selected_values)))
        cfg=applicable_setup(raw,base)
        changed=build(cfg)
        observed={int(v)+j for op in changed.ops for v,n in op[2] for j in range(n)}
        unused={int(name.split('.')[1]) for name,op in zip(changed.names,changed.ops)
                if name in selected and not any(int(v)+j in observed for v,n in op[3] for j in range(n))}
        unused &= selected_values
        if not unused:break
        selected_values-=unused
    else:raise AssertionError('Constant pruning did not converge')
    placed=np.array([selected[name] if name in selected and int(name.split('.')[1]) in selected_values
                     else named[name] for name in changed.names],dtype=np.int64)
    schedule=Scheduler(changed)
    units=np.array([placed[rows[0][0]]-rows[0][1] for rows in changed.units],dtype=np.int64)
    assert np.array_equal(placed,schedule.op_times(units))
    assert np.all(units[schedule.dests]>=units[schedule.sources]+schedule.lags)
    usage=np.zeros((cycles+33,len(RESOURCE_CAPACITY)),dtype=np.int64)
    for u,t in enumerate(units):usage[t:t+33]+=schedule.usage[u]
    assert np.all(usage<=RESOURCE_CAPACITY)
    bases,audit=allocate(changed,placed)
    report=dict(candidate=index,budget=budget,engine=engine,cycles=cycles,allocated=bases is not None,
                converted=len(selected_values),static_bundles=static_size(changed,cycles,placed),
                removed_operations=sorted(set(graph.names)-set(changed.names)),
                source=str(source),**counts(changed),**audit)
    directory=output/f'candidate_{index:03}';directory.mkdir(parents=True,exist_ok=True)
    for name,value in (('config',changed.config),('warm',report),('selection',selected),
                       ('semantics',verify_semantics(changed,(0,1)))):
        (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    if bases is not None:
        np.savez_compressed(directory/'best.npz',unit_times=units,times=placed)
    print(json.dumps(report),flush=True)
    return directory,report


def run(job):
    index,cfg,source,output,trials,iterations=job
    directory=output/f'candidate_{index:03}'
    warm=np.load(directory/'best.npz');units=warm['unit_times'].copy();times=warm['times'].copy()
    warm_report=json.loads((directory/'warm.json').read_text())
    result=search(job)
    if result[1]>warm_report['cycles']:
        np.savez_compressed(directory/'best.npz',unit_times=units,times=times)
        (directory/'search.json').write_text(json.dumps(warm_report,indent=2)+'\n')
        result=index,warm_report['cycles']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--engine',choices=('flow','load'),default='flow')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    jobs=[];rows=[]
    for i,budget in enumerate((4,8,12,24,1000)):
        directory,row=prepare(args.source,args.output,i,budget,args.engine);rows.append(row)
        if row['allocated']:
            jobs.append((i,json.loads((directory/'config.json').read_text()),directory,args.output,32,750))
    (args.output/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
    results.sort(key=lambda row:row[1]);(args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
