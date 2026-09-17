"""Measure shallow dispatch, prefetch, and two-round address folding."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search, static_size


def configurations(source):
    base = json.loads((source/'config.json').read_text())
    original = build(base)
    forms = dict(base.get('scalar_overrides') or {})
    pattern = re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    for name, op in zip(original.names, original.ops):
        stem = name.rsplit('.lane',1)[0]
        if pattern.fullmatch(stem):
            forms[stem] = op[0] == 'alu'
    base = dict(base, scalar_overrides=forms)
    unchanged = build(base)
    assert unchanged.ops == original.ops and unchanged.names == original.names
    for rnd in (2,13):
        for groups in ([4,5,6], [4,5,6,8,9,10], [16,17,18,20,21,22]):
            for folded in (False,True):
                for move_select in (False,True):
                    config=dict(base,pc_address_pools=False,merge_chains=[],
                                **{f'dispatch{rnd}_groups':groups})
                    if rnd==2:
                        config['prefetch5_groups']=[k for k in base['prefetch5_groups'] if k not in groups]
                    if folded:
                        config['fold_path3_groups']=[[rnd,k] for k in groups]
                    if move_select:
                        config['prefetch_madd_groups']=[]
                    yield dict(round=rnd,groups=groups,folded=folded,move_select=move_select),config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=350)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    jobs,screen=[],[]
    for description,raw in configurations(args.source):
        config=balance(raw,args.source,0)
        graph=build(config)
        stats=counts(graph)
        if static_size(graph,stats['bound'])>12000:
            continue
        verify_semantics(graph,(0,1))
        index=len(jobs)
        screen.append(dict(candidate=index,**description,**stats))
        jobs.append((index,config,args.source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
