"""Retain a verified schedule while putting its existing pause into the graph."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import build,Scheduler,RESOURCE_CAPACITY,allocate,bootstrap_cycle,counts
from resource_bounds import analyze_graph
from search_compact import static_size


def main():
    source=ROOT/'results/interleaved_pc_914/candidate_001';output=Path(__file__).parent
    cfg=json.loads((source/'config.json').read_text());old=build(cfg)
    old_times=np.load(source/'best.npz')['times'];named=dict(zip(old.names,map(int,old_times)))
    old_ops=dict(zip(old.names,old.ops));pause=json.loads((source/'verification.json').read_text())['pause_cycle']
    graph=build(dict(cfg,pc_prologue=13));scheduler=Scheduler(graph)
    assert set(graph.names)==set(old.names)|{'initial.pause'}
    for name,op in zip(graph.names,graph.ops):
        if name!='initial.pause':assert op==old_ops[name]
    times=np.array([pause if name=='initial.pause' else named[name] for name in graph.names],dtype=np.int64)
    units=np.array([times[rows[0][0]]-rows[0][1] for rows in graph.units],dtype=np.int64)
    assert np.array_equal(times,scheduler.op_times(units))
    assert np.all(units[scheduler.dests]>=units[scheduler.sources]+scheduler.lags)
    cycles=int(times.max())+1;usage=np.zeros((cycles+33,6),dtype=np.int64)
    for u,t in enumerate(units):usage[t:t+33]+=scheduler.usage[u]
    assert np.all(usage<=RESOURCE_CAPACITY)
    bases,audit=allocate(graph,times);assert bases is not None,audit
    report=dict(source=str(source),cycles=cycles,pause_cycle=pause,
                bootstrap_cycle=bootstrap_cycle(graph,times),static_bundles=static_size(graph,cycles,times),
                all_existing_operations_and_times_unchanged=True,**counts(graph),**audit)
    for name,value in (('config',graph.config),('warm',report),('bounds',analyze_graph(graph))):
        (output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(output/'best.npz',times=times,unit_times=units)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
