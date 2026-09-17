"""Use interleaved child rows without permuting unrelated dispatch tables."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    source=ROOT/'results/compact_914_pc';output=Path(__file__).parent
    base=json.loads((source/'config.json').read_text());jobs=[];rows=[]
    for early in ([],[24,28],[16,20,24,28]):
        for late in ([20,24],[16,20,24],[8,12,16,20,24],list(range(0,28,4))):
            raw=dict(base,natural_pc_order=True,pair_even_odd_order=True,
                     early_pair_groups=early,late_pair_groups=late)
            cfg=applicable_setup(raw,base);cfg=balance(cfg,source,0)
            graph=build(cfg);bounds=analyze_graph(graph);index=len(jobs)
            directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
            row=dict(candidate=index,early=early,late=late,resource_bound=bounds['bound'],**counts(graph))
            for name,value in (('config',cfg),('bounds',bounds),('parameters',row),
                               ('semantics',verify_semantics(graph,(0,1)))):
                (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            rows.append(row);print(json.dumps(row),flush=True)
            jobs.append((index,cfg,source,output,32,750))
    (output/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1]);(output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
