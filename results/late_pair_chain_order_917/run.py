"""Choose lookup chains using saved execution order and measured gaps."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def chains(regions,times,length,stretch,offset):
    result=[]
    i=offset
    while i<len(regions):
        batch=[regions[i]]
        first=int(times[regions[i]['start']])
        while i+len(batch)<len(regions) and len(batch)<length:
            candidate=regions[i+len(batch)]
            if stretch is not None and int(times[candidate['start']])-first-8*len(batch)>stretch:
                break
            batch.append(candidate)
        if len(batch)>1:
            result.append([[14,r['groups'][0]] for r in batch])
        i+=len(batch)
    return result


def main():
    output=Path(__file__).resolve().parent
    jobs,screen=[],[]
    for candidate in (12,36,37,42):
        source=Path('results/late_pair_rows_917')/f'candidate_{candidate:03}'
        base=json.loads((source/'config.json').read_text())
        graph=build(base)
        times=np.load(source/'best.npz')['times']
        regions=sorted([r for r in graph.regions if r['round']==14],key=lambda r:times[r['start']])
        seen=set()
        for length,stretch,offset in ((2,2,0),(2,4,0),(2,None,0),(2,None,1),(3,4,0),(3,8,0),(4,6,0),(4,None,0)):
            added=chains(regions,times,length,stretch,offset)
            signature=json.dumps(added)
            if not added or signature in seen:continue
            seen.add(signature)
            config=dict(base,merge_chains=base['merge_chains']+added)
            proposed=build(config)
            Scheduler(proposed)
            verify_semantics(proposed,(0,1))
            index=len(jobs)
            screen.append(dict(candidate=index,source=str(source),length=length,stretch=stretch,
                               offset=offset,chains=added,**counts(proposed)))
            jobs.append((index,config,source,output,20,500))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
