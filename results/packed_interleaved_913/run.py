"""Pack four PC banks into holes while retaining the exact 913-cycle schedule."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import build,Scheduler,RESOURCE_CAPACITY,allocate,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import static_size


def main():
    source=ROOT/'results/compact_913';output=Path(__file__).parent
    cfg=json.loads((source/'config.json').read_text());old=build(cfg)
    graph=build(dict(cfg,pack_interleaved_tables=True));schedule=Scheduler(graph)
    saved=np.load(source/'best.npz');times=saved['times'];units=saved['unit_times']
    assert graph.names==old.names and graph.units==old.units
    assert counts(graph)['engines']==counts(old)['engines']
    assert np.array_equal(times,schedule.op_times(units))
    assert np.all(units[schedule.dests]>=units[schedule.sources]+schedule.lags)
    cycles=int(times.max())+1;usage=np.zeros((cycles+33,6),dtype=np.int64)
    for u,t in enumerate(units):usage[t:t+33]+=schedule.usage[u]
    assert np.all(usage<=RESOURCE_CAPACITY)
    bases,audit=allocate(graph,times);assert bases is not None,audit
    changes=[name for name,a,b in zip(graph.names,old.ops,graph.ops) if a!=b]
    assert len(changes)==2,changes
    report=dict(source=str(source),cycles=cycles,static_bundles=static_size(graph,cycles,times),
                changed_operations=changes,all_operation_times_unchanged=True,
                static_bundle_reduction=static_size(old,cycles,times)-static_size(graph,cycles,times),
                **counts(graph),**audit)
    for name,value in (('config',graph.config),('packing',report),('bounds',analyze_graph(graph)),
                       ('semantics',verify_semantics(graph,(0,1)))):
        (output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(output/'best.npz',times=times,unit_times=units)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
