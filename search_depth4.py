"""Move selected lookups from depth three to four and prefetch depth five."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np

from balance_resources import balance
from optimize import build,counts,verify_semantics
from search_compact import search,static_size


def configurations(source):
    base=json.loads((source/'config.json').read_text())
    original=build(base)
    times=np.load(source/'best.npz')['times']
    named=dict(zip(original.names,map(int,times)))
    old_region={(r['round'],k):int(times[r['start']]) for r in original.regions for k in r['groups']}
    forms=dict(base.get('scalar_overrides') or {})
    pattern=re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    for name,op in zip(original.names,original.ops):
        stem=name.rsplit('.lane',1)[0]
        if pattern.fullmatch(stem): forms[stem]=op[0]=='alu'
    base=dict(base,scalar_overrides=forms)
    retained=build(base)
    assert retained.names==original.names and retained.ops==original.ops
    grand=base['prefetch5_groups']
    singles=[r['groups'][0] for r in original.regions if r['round']==3 and r['width']==1]
    selections=[grand[:4],grand[:8],grand,singles,list(range(16))]
    for chosen in selections:
        for pools in (False,True):
            for keep in ([5],[5,6],[5,7],[4,5,6,7]):
                for flow in (False,True):
                    config=dict(base,dispatch4_groups=chosen,gather_dispatch4=True,prefetch4=True,width4=1,
                                prefetch5_groups=[k for k in grand if k not in chosen],pc_address_pools=pools,
                                heap_keep_levels=keep,lane_allocation_trials=16)
                    config['merge_chains']=[chain for chain in base['merge_chains']
                                            if all(r!=3 or k not in chosen for r,k in chain)]
                    if flow: config['prefetch_madd_groups']=[]
                    graph=build(config)
                    order=[]
                    for region in graph.regions:
                        r,k=region['round'],region['groups'][0]
                        priority=named[f'r4.g{k}.prefetched_node']+3 if r==4 else old_region[r,k]
                        order.append((priority,r,k))
                    config['temp_region_order']=[[r,k] for _,r,k in sorted(order)]
                    yield config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    configs=[];screen=[]
    for raw in configurations(args.source):
        config=balance(raw,args.source,policy=0)
        graph=build(config);stats=counts(graph)
        if static_size(graph,stats['bound'])>12000:continue
        semantic=verify_semantics(graph,(0,))
        configs.append(config)
        screen.append(dict(candidate=len(configs)-1,**stats,semantic=semantic))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    jobs=[(i,c,args.source,args.output,args.trials,args.iterations) for i,c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
