"""Small STORE-span changes around the verified 915 schedule."""
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
    source=Path('results/compact_915');output=Path(__file__).parent
    base=json.loads((source/'config.json').read_text())
    jobs=[]
    for index,(span,number,pairs) in enumerate(itertools.product((2,3),(1,3,6),((),(20,24)))):
        cfg=dict(base,dispatch_spans=[(3,k,span) for k in base['prefetch5_groups'][:number]],
                 heap_keep_levels=[4,5,6],late_pair_groups=list(pairs),lane_allocation_trials=16)
        cfg=balance(cfg,source,0)
        graph=build(cfg);bound=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
        directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bound),('semantics',semantics),
                           ('parameters',dict(span=span,number=number,late_pairs=pairs))):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((index,cfg,source,output,24,500))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
