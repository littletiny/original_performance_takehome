"""Build initial PC-offset vectors from an existing address bank with MADD."""
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
    parser.add_argument('--output',type=Path,default=Path('results/pc_offset_madd_914'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path('results/compact_914');base=json.loads((source/'config.json').read_text())
    original=build(base);named=dict(zip(original.names,map(int,np.load(source/'best.npz')['times'])))
    c=[row[1] for row in frozen.HASH_STAGES]
    critical={f'broadcast.{x&0xffffffff}' for x in
              (c[0],c[1],c[2]+c[3],c[2]<<9,c[4],c[5],4097,19,33,33<<9,9,16)}
    jobs=[];screen=[]
    profiles=[(False,[],True),(True,[],True),(True,[24,28],True),
              (True,[12,16,20,24,28],True),(True,[],False)]
    for all_order,early,madd in profiles:
        raw=dict(base,pair_even_odd_order=all_order,all_even_odd_order=all_order,
                 early_pair_groups=early,pc_offset_madd=madd,lane_allocation_trials=16)
        probe=build(dict(raw,memory_vectors=[],memory_vector_order=[],constant_expressions=None))
        readers={}
        for name,op in zip(probe.names,probe.ops):
            tag=op[4]
            fallback=named.get(f'r{tag[0]}.g{tag[1]}.h1',1000)-2
            t=named.get(name,named.get(name.rsplit('.lane',1)[0],fallback))
            for value,size in op[2]:
                for j in range(size):readers[int(value)+j]=min(readers.get(int(value)+j,10000),t)
        uniform={name:min(readers.get(int(op[3][0][0])+j,10000) for j in range(8))
                 for name,op in zip(probe.names,probe.ops)
                 if op[1][0]=='vbroadcast' or name.startswith('derive.')}
        existing=[name for name in base['memory_vectors'] if name in uniform]
        exclude=critical|{'root.raw'}|{f'{prefix}.{row["delta"]}'
                                     for row in probe.pc_madd_offsets for prefix in ('broadcast','derive')}
        additions=sorted((name for name in uniform if name not in existing and name not in exclude),
                         key=lambda name:(-uniform[name],name))
        for extra in ((0,4,8,12) if madd else (0,)):
            selected=existing+additions[:extra]
            order=sorted(selected,key=lambda name:(uniform[name],name))
            setup=dict(base,memory_vectors=selected,memory_vector_order=order)
            cfg=applicable_setup(raw,setup);cfg=balance(cfg,source,0)
            graph=build(cfg);bounds=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
            index=len(jobs);directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
            row=dict(candidate=index,all_order=all_order,early=early,madd=madd,extra=extra,
                     additions=additions[:extra],resource_bound=bounds['bound'],**counts(graph))
            screen.append(row)
            for name,value in (('config',cfg),('bounds',bounds),('semantics',semantics),
                               ('parameters',row),('pc_offsets',graph.pc_madd_offsets)):
                (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            jobs.append((index,cfg,source,args.output,32,750))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
