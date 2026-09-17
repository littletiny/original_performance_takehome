"""Move the PC MADD addend through a timed memory-broadcast row."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search


def main():
    source=Path('results/pc_offset_madd_914/candidate_002');output=Path(__file__).parent
    base=json.loads((source/'config.json').read_text());jobs=[];screen=[]
    for position,policy in itertools.product((0,1,2,3,4,5,12),(0,1)):
        order=list(base['memory_vector_order']);order.insert(position,'broadcast.2462')
        cfg=dict(base,memory_vectors=base['memory_vectors']+['broadcast.2462'],
                 memory_vector_order=order)
        cfg=balance(cfg,source,policy);graph=build(cfg);bounds=analyze_graph(graph)
        semantics=verify_semantics(graph,(0,1))
        index=len(jobs);directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        row=dict(candidate=index,position=position,policy=policy,resource_bound=bounds['bound'],**counts(graph))
        screen.append(row)
        for name,value in (('config',cfg),('bounds',bounds),('semantics',semantics),('parameters',row)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((index,cfg,source,output,32,750))
        print(json.dumps(row),flush=True)
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
