"""Release/tail resource lower bounds for a fixed logical schedule graph."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from optimize import build,Scheduler,ENGINES,CAPACITY,counts


def _analyze_graph(graph,scheduler,source):
    early=np.zeros(len(graph.units),dtype=np.int64)
    for unit in scheduler.order:
        for child,lag in scheduler.children[unit]:
            early[child]=max(early[child],early[unit]+lag)
    release=scheduler.op_times(early)
    tail=scheduler.tail[scheduler.op_units]-scheduler.op_offsets
    assert np.all(tail>=0)
    evidence={}
    for engine,capacity in [*zip(ENGINES,CAPACITY),('arithmetic',60)]:
        hist=np.zeros((int(release.max())+1,int(tail.max())+1),dtype=np.int64)
        for i,op in enumerate(graph.ops):
            weight=(8 if op[0]=='valu' else int(op[0]=='alu')) if engine=='arithmetic' else int(op[0]==engine)
            if weight:hist[release[i],tail[i]]+=weight
        work=hist[::-1,::-1].cumsum(0).cumsum(1)[::-1,::-1]
        bounds=np.arange(len(hist))[:,None]+np.arange(hist.shape[1])[None,:]+(work+capacity-1)//capacity
        bounds[work==0]=0
        r,t=np.unravel_index(bounds.argmax(),bounds.shape)
        evidence[engine]=dict(bound=int(bounds[r,t]),release=int(r),tail=int(t),
                              work=int(work[r,t]),capacity=int(capacity))
    critical=int((early+scheduler.durations).max())+1
    bound=max(critical,*[v['bound'] for v in evidence.values()])
    report=dict(source=str(source),counts=counts(graph),bound=bound,
                critical_path_bound=critical,resource_windows=evidence,
                formula='C >= release + tail + ceil(work / capacity)',
                scope=('Fixed SSA graph and compound units; scratch ignored. Bootstrap is not included; '
                       'the initial pause is included. Not a global task lower bound.' if graph.config.get('pc_prologue') else
                       'Fixed SSA graph and compound units; scratch ignored. Bootstrap and pause are not included. Not a global task lower bound.'))
    signature=json.dumps([graph.ops,graph.units,graph.control,graph.sizes],separators=(',',':'))
    report['graph_sha256']=hashlib.sha256(signature.encode()).hexdigest()
    return report


def analyze_graph(graph,source='unsaved graph'):
    """Evaluate a proposed graph before spending work on an allocated schedule."""
    return _analyze_graph(graph,Scheduler(graph),source)


def analyze(source):
    graph=build(json.loads((source/'config.json').read_text()))
    scheduler=Scheduler(graph)
    report=_analyze_graph(graph,scheduler,source)
    if (source/'best.npz').exists():
        schedule=np.load(source/'best.npz')
        times=schedule['times'];units=schedule['unit_times']
        assert len(times)==len(graph.ops)
        assert np.array_equal(times,scheduler.op_times(units))
        assert np.all(units[scheduler.dests]>=units[scheduler.sources]+scheduler.lags)
        cycles=int(times.max())+1
        assert report['bound']<=cycles,(report['bound'],cycles)
        report['saved_schedule_cycles']=cycles
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[analyze(source) for source in args.sources]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ('counts','graph_sha256','scope')}),flush=True)


if __name__=='__main__':
    main()
