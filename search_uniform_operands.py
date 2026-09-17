"""Measure direct scalar reads of uniform operands and removed broadcasts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.source/'config.json').read_text())
    families=((),('h2.a',),('h2.b',),('h6.a',),('h6.b',),
              ('h2.a','h2.b'),('h6.a','h6.b'),('h2.a','h6.a'))
    jobs,screen=[],[]
    for labels in families:
        for roots in (False,True):
            cfg=dict(base,share_uniform_operands=True,scalar_constant_labels=list(labels),
                     scalar_setup=['tree.shallow.bias','tree.d3.bias0'] if 'h6.a' in labels else [],
                     scalar_root_groups='all' if roots else [])
            try:
                cfg=balance(cfg,args.source,0)
            except AssertionError as error:
                screen.append(dict(labels=labels,roots=roots,rejected=str(error)))
                continue
            g=build(cfg);verify_semantics(g,(0,1))
            index=len(jobs)
            screen.append(dict(candidate=index,labels=labels,roots=roots,**counts(g)))
            jobs.append((index,cfg,args.source,args.output,20,500))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
