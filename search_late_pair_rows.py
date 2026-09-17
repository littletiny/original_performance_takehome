"""Measure adjacent-child row stores with all physical padding accounted for."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def configurations(source):
    base=json.loads((source/'config.json').read_text())
    original=build(base)
    times=np.load(source/'best.npz')['times']
    regions=[r for r in original.regions if r['round']==14]
    pairs=[r['groups'][0] for r in regions if r['width']==2]
    execution=sorted(regions,key=lambda r:times[r['start']])
    order={r['groups'][0]:i for i,r in enumerate(execution)}
    selections=[pairs[:2],pairs[-2:],pairs[:4],pairs]
    for selected in selections:
        for buffers in (1,2):
            for chain in (0,2,4):
                for loads in (False,True):
                    config=dict(base,late_pair_groups=selected,late_pair_buffers=buffers)
                    if chain:
                        added=[]
                        for start in range(0,len(regions),chain):
                            batch=regions[start:start+chain]
                            if len(batch)>1:
                                added.append([[14,r['groups'][0]] for r in batch])
                        config['merge_chains']=base['merge_chains']+added
                    if loads:
                        addresses=set(config.get('force_load_scalars',()))
                        for buffer in range(buffers):
                            address=2054+48*buffer
                            addresses.update(address+24*s+2*j for s in range(2) for j in range(8))
                            addresses.update(address+j for j in range(0,48,8))
                        config['force_load_scalars']=sorted(addresses)
                    if not chain:
                        config['late_pair_order']=sorted(selected,key=order.get)
                    yield dict(groups=selected,buffers=buffers,chain=chain,load_pointers=loads),config


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
