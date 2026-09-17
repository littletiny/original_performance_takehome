"""Spend free FLOW capacity around the verified 914-cycle schedule."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search


def main():
    source=Path('results/scheduled_constants_915/candidate_008');output=Path(__file__).parent
    base=json.loads((source/'config.json').read_text());jobs=[];screen=[]
    for index,(prefixes,madds,path_round) in enumerate(itertools.product(
            ([],[0,1]),(12,8,4,0),(None,1,12))):
        cfg=dict(base,early_prefix_groups=prefixes,fold_path4_groups=prefixes,
                 prefetch_madd_groups=base['prefetch_madd_groups'][:madds])
        if path_round is not None:
            chosen={(path_round,k) for k in range(8)}
            cfg.update(path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)
                                                        if (r,k) not in chosen])
        cfg=balance(cfg,source,0);graph=build(cfg)
        bound=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
        row=dict(candidate=index,prefixes=prefixes,madds=madds,path_round=path_round,
                 resource_bound=bound['bound'],**bound['counts'])
        directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bound),('semantics',semantics),('parameters',row)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        screen.append(row);print(json.dumps(row),flush=True)
        jobs.append((index,cfg,source,output,24,600))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
