"""Rewrite constants only through operands ready in a verified warm schedule."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from constant_graph import expressions
from optimize import build,counts,Scheduler,allocate,verify_semantics,RESOURCE_CAPACITY
from search_compact import search,static_size


def replacements(graph,times,policy,budget):
    known=graph.scalar_constants
    writers={int(v)+j:i for i,op in enumerate(graph.ops) for v,n in op[3] for j in range(n)}
    zero={int(v)+j for v in graph.initial_zero for j in range(graph.sizes[v.vid])}
    available={n:v for n,v in known.items() if int(v) in writers or int(v) in zero}
    ready={n:int(times[writers[int(v)]])+1 if int(v) in writers else 0 for n,v in available.items()}
    numeric={int(v):n for n,v in available.items()}
    last={n:0 for n in available}
    for i,op in enumerate(graph.ops):
        for ref,size in op[2]:
            for j in range(size):
                if int(ref)+j in numeric:
                    n=numeric[int(ref)+j];last[n]=max(last[n],int(times[i]))
    options=expressions(tuple(sorted(available)))
    changes=[]
    for i,(name,old) in enumerate(zip(graph.names,graph.ops)):
        if not name.startswith('constant.') or old[0]!='alu':continue
        value=int(name.split('.')[1]);t=int(times[i])
        if value not in available or old[3]!=[(available[value],1)]:continue
        def key(expression):
            _,a,b=expression
            extension=max(0,t-last[a])+max(0,t-last[b])
            release=max(ready[a],ready[b])
            return (release,extension) if policy=='early' else (extension,release)
        old_expression=(old[1][0],*(numeric[int(v)] for v in old[1][2:]))
        viable=[expr for expr in options[value] if max(ready[expr[1]],ready[expr[2]])<=t]
        best=min(viable,key=key,default=old_expression)
        if key(best)<key(old_expression):
            gain=key(old_expression)[0]-key(best)[0]
            changes.append((-gain,name,value,best))
    return {str(value):expression for _,_,value,expression in sorted(changes)[:budget]}


def run(job):
    result=search(job)
    index,_,_,output,_,_=job
    directory=output/f'candidate_{index:03}'
    warm=json.loads((directory/'warm.json').read_text())
    if warm['allocated'] and result[1]>warm['cycles']:
        shutil.copyfile(directory/'warm.npz',directory/'best.npz')
        (directory/'search.json').write_text(json.dumps(warm,indent=2)+'\n')
        result=(index,warm['cycles'])
    return result


def main():
    output=Path('results/scheduled_constants_915');output.mkdir(exist_ok=True)
    jobs=[]
    sources=(Path('results/compact_915'),Path('results/prefetch_pairs_915/candidate_000'))
    for source,policy,budget in itertools.product(sources,('early','lifetime'),(8,32,10000)):
        base=json.loads((source/'config.json').read_text())
        graph=build(base);times=np.load(source/'best.npz')['times']
        named=dict(zip(graph.names,map(int,times)))
        chosen=replacements(graph,times,policy,budget)
        cfg=dict(base,constant_expressions=chosen,lane_allocation_trials=16)
        new=build(cfg);scheduler=Scheduler(new)
        new_times=np.array([named[name] for name in new.names],dtype=np.int64)
        units=[]
        for unit in new.units:
            starts={int(new_times[i])-offset for i,offset in unit}
            assert len(starts)==1
            units.append(starts.pop())
        units=np.asarray(units,dtype=np.int64)
        assert np.all(units[scheduler.dests]>=units[scheduler.sources]+scheduler.lags)
        cycles=int(new_times.max())+1
        use=np.zeros((cycles,len(RESOURCE_CAPACITY)),dtype=np.int64)
        for u,start in enumerate(units):
            length=int(scheduler.durations[u])+1
            use[start:start+length]+=scheduler.usage[u,:length]
        assert np.all(use<=RESOURCE_CAPACITY)
        bases,audit=allocate(new,new_times)
        index=len(jobs);directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        warm=dict(candidate=index,trial=-1,cycles=cycles,allocated=bases is not None,
                  static_bundles=static_size(new,cycles,new_times),policy=policy,budget=budget,
                  source=str(source),replacements=len(chosen),**counts(new),**audit)
        for name,data in (('config',new.config),('warm',warm),('semantics',verify_semantics(new,(0,1)))):
            (directory/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
        np.savez_compressed(directory/'warm.npz',unit_times=units,times=new_times)
        print(json.dumps(warm),flush=True)
        jobs.append((index,cfg,source,output,32,600))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
