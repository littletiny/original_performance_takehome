"""Screen vector stores of grandchild rows with validated buffer reuse."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from search_compact import search,static_size


def configurations(source):
    base=json.loads((source/'config.json').read_text())
    grand=base['prefetch5_groups']
    for buffers in (1,2,3):
        for number in (4,8,len(grand)):
            selected=[grand[i*len(grand)//number] for i in range(number)]
            for keep in ([5],[5,6],[5,7],[4,5,6,7]):
                yield dict(base,grand_row_groups=selected,grand_row_buffers=buffers,
                           heap_keep_levels=keep,lane_allocation_trials=16)
    for buffers in (1,2,3):
        for remaining in (0,8):
            for keep in ([5],[4,5,6,7]):
                yield dict(base,grand_row_groups=grand,grand_row_buffers=buffers,
                           heap_keep_levels=keep,lane_allocation_trials=16,
                           prefetch_madd_groups=[[4,k] for k in range(remaining)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    configs=[]
    screen=[]
    for raw in configurations(args.source):
        config=balance(raw,args.source,policy=0)
        graph=build(config)
        stats=counts(graph)
        assert static_size(graph,stats['bound'])<=12000
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
