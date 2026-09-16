"""Measure path folding, FLOW prefetch choices, and sparse depth-5 dispatch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search, static_size


def configurations(source):
    base=json.loads((source/'config.json').read_text())
    graph=build(base)
    pattern=re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    forms=dict(base.get('scalar_overrides') or {})
    for name,op in zip(graph.names,graph.ops):
        stem=name.rsplit('.lane',1)[0]
        if pattern.fullmatch(stem):
            forms[stem]=op[0]=='alu'
    base=dict(base,scalar_overrides=forms)
    retained=build(base)
    assert retained.names==graph.names and retained.ops==graph.ops
    grand=base['prefetch5_groups']
    for remaining in (0,4,8,12,16):
        for groups in ([],grand[:4],grand[:8],grand,list(range(12))):
            yield dict(base,prefetch_madd_groups=[[4,k] for k in range(remaining)],fold_path4_groups=groups)
    times=np.load(source/'best.npz')['times']
    named=dict(zip(graph.names,map(int,times)))
    ordered=sorted(grand,key=lambda k:named[f'r5.g{k}.h1'])
    for number in (1,2,4):
        for selected in (ordered[:number],ordered[-number:]):
            kept=[k for k in grand if k not in selected]
            for remaining in (0,8,16):
                config=dict(base,dispatch5_groups=selected,prefetch5_groups=kept,
                            prefetch_madd_groups=[[4,k] for k in range(remaining)])
                yield config
                yield dict(config,fold_path4_groups=kept[:4])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=400)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    configs=[]
    for raw in configurations(args.source):
        config=balance(raw,args.source,policy=0)
        graph=build(config)
        stats=counts(graph)
        if static_size(graph,stats['bound'])>12000:
            continue
        verify_semantics(graph,(0,))
        configs.append(config)
    jobs=[(i,c,args.source,args.output,args.trials,args.iterations) for i,c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
