"""Keep existing PCs and match packed-row reuse to the measured producer order."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    output=Path(__file__).parent;source=ROOT/'results/compact_914_pc'
    base=json.loads((source/'config.json').read_text())
    original=build(base);times=dict(zip(original.names,map(int,np.load(source/'best.npz')['times'])))
    profiles=[[[3,7]],[[10,11]],[[3,7,10]],[[10,11,14]]]
    jobs=[];screen=[]
    for groups in profiles:
        members={k for group in groups for k in group}
        for order in (sorted(members),sorted(members,key=lambda k:times[f'r3.g{k}.dispatch.jump0'])):
            for natural in (False,True):
                raw=dict(base,quad_dispatch_groups=groups,quad_row_buffers=1,quad_row_order=order,
                         natural_pc_order=natural,pair_even_odd_order=True,all_even_odd_order=not natural,
                         prefetch_madd_groups=[p for p in base['prefetch_madd_groups'] if p[1] not in members])
                cfg=applicable_setup(raw,base);cfg=balance(cfg,source,0)
                graph=build(cfg);bounds=analyze_graph(graph);index=len(jobs)
                directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
                row=dict(candidate=index,groups=groups,order=order,natural=natural,
                         resource_bound=bounds['bound'],**counts(graph))
                semantics=verify_semantics(graph,(0,1))
                for name,value in (('config',cfg),('bounds',bounds),('parameters',row),('semantics',semantics)):
                    (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
                screen.append(row);print(json.dumps(row),flush=True)
                jobs.append((index,cfg,source,output,32,750))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1]);(output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
