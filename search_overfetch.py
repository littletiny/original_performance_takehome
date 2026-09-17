"""Use an implicit VLOAD lane offset to remove deep address compensation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--trials',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=500)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.source/'config.json').read_text())
    jobs,screen=[],[]
    for count in (8,16,32):
        for location in ('head','spread'):
            groups=list(range(count)) if location=='head' else list(range(0,32,32//count))
            if count==32 and location=='spread':continue
            for transfer in ('none','madd','fold'):
                raw=dict(base,overfetch_groups=groups)
                if transfer in ('madd','fold'):raw['prefetch_madd_groups']=[]
                if transfer=='fold':raw['fold_path4_groups']=list(range(min(count,24)))
                # Address changes may eliminate or change the construction of
                # constants selected for memory replication by the source.
                probe=build(dict(raw,memory_vectors=[],memory_vector_order=[]))
                available={name for name,op in zip(probe.names,probe.ops)
                           if op[1][0]=='vbroadcast' or name.startswith('derive.')}
                raw['memory_vectors']=[name for name in base.get('memory_vectors',()) if name in available]
                raw['memory_vector_order']=[name for name in base.get('memory_vector_order',()) if name in available]
                for policy in (0,1):
                    config=balance(raw,args.source,policy)
                    graph=build(config);Scheduler(graph);verify_semantics(graph,(0,1))
                    index=len(jobs)
                    screen.append(dict(candidate=index,groups=groups,transfer=transfer,policy=policy,**counts(graph)))
                    jobs.append((index,config,args.source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
