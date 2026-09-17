"""Transfer selected constant/root broadcasts to STORE fills plus VLOAD."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def configurations(source):
    config=json.loads((source/'config.json').read_text())
    graph=build(config)
    times=np.load(source/'best.npz')['times']
    broadcasts=[name for name,op in zip(graph.names,graph.ops) if op[1][0]=='vbroadcast']
    derives=[name for name in graph.names if name.startswith('derive.')]
    named=dict(zip(graph.names,map(int,times)))
    families={
        'constants':[n for n in broadcasts if n.startswith('broadcast.')],
        'nodes':[n for n in broadcasts if n.startswith(('root.','table.'))],
        'all_broadcasts':broadcasts,
        'all_uniform':broadcasts+derives,
        'late_half':sorted(broadcasts,key=named.get)[len(broadcasts)//2:],
    }
    for family,selected in families.items():
        for buffers in (1,2,4):
            for order in ('build','execution'):
                config2=dict(config,memory_vectors=selected,memory_vector_buffers=buffers,
                             memory_vector_order=sorted(selected,key=named.get) if order=='execution' else [])
                yield dict(family=family,buffers=buffers,order=order),config2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    jobs,screen=[],[]
    for description,raw in configurations(args.source):
        config=balance(raw,args.source,0)
        graph=build(config)
        Scheduler(graph)
        verify_semantics(graph,(0,1))
        index=len(jobs)
        screen.append(dict(candidate=index,**description,**counts(graph)))
        jobs.append((index,config,args.source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
