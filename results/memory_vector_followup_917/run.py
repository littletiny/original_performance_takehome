"""Keep critical startup broadcasts in registers; prefill later vectors early."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def main():
    source=Path('results/compact_917')
    output=Path(__file__).resolve().parent
    config=json.loads((source/'config.json').read_text())
    graph=build(config)
    times=np.load(source/'best.npz')['times']
    first={}
    for i,(name,op) in enumerate(zip(graph.names,graph.ops)):
        if op[1][0]!='vbroadcast' and not name.startswith('derive.'):continue
        words=set(range(int(op[3][0][0]),int(op[3][0][0])+8))
        consumers=[int(times[j]) for j,child in enumerate(graph.ops)
                   if any(int(v)+lane in words for v,size in child[2] for lane in range(size))]
        first[name]=min(consumers)
    jobs,screen=[],[]
    for threshold in (40,100,300):
        selected=sorted((name for name,t in first.items() if t>=threshold),key=first.get)
        for buffers in (1,2):
            for lead in (0,16,64):
                cfg=dict(config,memory_vectors=selected,memory_vector_order=selected,
                         memory_vector_buffers=buffers,memory_vector_lead=lead)
                cfg=balance(cfg,source,0)
                proposed=build(cfg)
                Scheduler(proposed)
                verify_semantics(proposed,(0,1))
                index=len(jobs)
                screen.append(dict(candidate=index,threshold=threshold,buffers=buffers,lead=lead,
                                   names=selected,**counts(proposed)))
                jobs.append((index,cfg,source,output,16,500))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
