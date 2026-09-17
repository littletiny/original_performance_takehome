"""Place compensating vector operations near the newly scalar late work."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search


def main():
    source=Path('results/compact_917')
    output=Path(__file__).resolve().parent
    base=json.loads((source/'config.json').read_text())
    original=build(base)
    times=np.load(source/'best.npz')['times']
    regions=sorted([r for r in original.regions if r['round']==14],key=lambda r:times[r['start']])
    order={r['groups'][0]:i for i,r in enumerate(regions)}
    all_pairs=[0,4,8,12,16,20,24]
    jobs,screen=[],[]
    for groups,buffers in ((all_pairs[-2:],1),(all_pairs[:4],1),(all_pairs,1),(all_pairs,2)):
        for policy in ('late','pair_local','pair_final','final_round'):
            cfg=dict(base,late_pair_groups=groups,late_pair_buffers=buffers,
                     late_pair_order=sorted(groups,key=order.get))
            cfg=balance(cfg,source,policy)
            graph=build(cfg)
            verify_semantics(graph,(0,1))
            index=len(jobs)
            screen.append(dict(candidate=index,groups=groups,buffers=buffers,policy=policy,**counts(graph)))
            jobs.append((index,cfg,source,output,24,500))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
