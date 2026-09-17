"""Use the fixed-table prefix for a scheduled pause and constant FLOW work."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from balance_resources import balance
from optimize import build,counts,frozen,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/prologue_constants_914'))
    parser.add_argument('--trials',type=int,default=32)
    parser.add_argument('--iterations',type=int,default=750)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    c=[row[1] for row in frozen.HASH_STAGES]
    profiles=[[],[19],[4097],[c[0]],[19,4097],[c[0],4097],
              [c[0],c[1],19,4097],[c[0],c[1],c[5],19,4097,16896,9]]
    jobs=[];screen=[]
    for source in (Path('results/compact_914_pc'),Path('results/interleaved_pc_914/candidate_001')):
        base=json.loads((source/'config.json').read_text());old=build(base)
        times=np.load(source/'best.npz')['times']
        arithmetic=[int(name.split('.')[1]) for _,name in sorted(
                    (int(t),name) for name,op,t in zip(old.names,old.ops,times)
                    if name.startswith('constant.') and op[0]=='alu')]
        for constants in profiles+[arithmetic[:4],arithmetic[:8],arithmetic[:12]]:
            raw=dict(base,pc_prologue=13,flow_constants=constants)
            cfg=applicable_setup(raw,base);cfg=balance(cfg,source,0)
            graph=build(cfg);bounds=analyze_graph(graph);index=len(jobs)
            row=dict(candidate=index,source=str(source),flow_constants=constants,
                     resource_bound=bounds['bound'],**counts(graph))
            directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
            for name,value in (('config',cfg),('bounds',bounds),('parameters',row),
                               ('semantics',verify_semantics(graph,(0,1)))):
                (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            screen.append(row);print(json.dumps(row),flush=True)
            jobs.append((index,cfg,source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1]);(args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
