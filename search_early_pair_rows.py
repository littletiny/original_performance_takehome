"""Use an even/odd handler order to keep early branch bits contiguous."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/early_pair_rows_914'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sources=(Path('results/compact_914'),Path('results/overfetch_levels_914/candidate_011'))
    pairs=[0,4,8,12,16,20,24,28]
    jobs=[];screen=[]
    for source,count,late in itertools.product(sources,(0,2,5,8),((),(20,24))):
        base=json.loads((source/'config.json').read_text())
        early=pairs[-count:] if count else []
        members={k for group in early for k in (group,group+1)}
        cfg=applicable_setup(dict(base,early_pair_groups=early,late_pair_groups=list(late),
                                  pair_even_odd_order=True,lane_allocation_trials=16,
                                  prefetch_madd_groups=[p for p in base['prefetch_madd_groups']
                                                       if p[0]!=4 or p[1] not in members]),base)
        cfg=balance(cfg,source,0);graph=build(cfg);bounds=analyze_graph(graph)
        semantics=verify_semantics(graph,(0,1))
        index=len(jobs);directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        row=dict(candidate=index,source=str(source),early=early,late=late,
                 resource_bound=bounds['bound'],**counts(graph))
        screen.append(row)
        for name,value in (('config',cfg),('bounds',bounds),('semantics',semantics),('parameters',row)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        jobs.append((index,cfg,source,args.output,32,750))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
