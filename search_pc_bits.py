"""Fuse final path updates into PC offsets backed by existing scalar pointers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--trials',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    base=json.loads((args.source/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    jobs,screen=[],[]
    for pools in ([2318],[2318,2382],[2318,2446],[2318,2510],[2318,2382,2446,2510]):
        for policy in (None,0,1):
            config=dict(base,pc_bit_pools=pools)
            if policy is not None:
                config=balance(config,args.source,policy)
            graph=build(config)
            verify_semantics(graph,(0,1))
            index=len(jobs)
            screen.append(dict(candidate=index,pools=pools,balance_policy=policy,**counts(graph)))
            jobs.append((index,config,args.source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
