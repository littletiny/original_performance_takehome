"""Choose overfetch independently per depth, then spend released FLOW slots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search


def applicable_setup(raw,source_config):
    probe=build(dict(raw,memory_vectors=[],memory_vector_order=[],constant_expressions=None))
    available=set(probe.names)
    cfg=dict(raw,memory_vectors=[n for n in source_config['memory_vectors'] if n in available],
             memory_vector_order=[n for n in source_config['memory_vector_order'] if n in available],
             constant_expressions=None)
    probe=build(cfg)
    named=dict(zip(probe.names,probe.ops));known=probe.scalar_constants
    recipes={}
    for target,expression in (source_config.get('constant_expressions') or {}).items():
        value=int(target);_,a,b=expression
        op=named.get(f'constant.{value}')
        if (op is not None and op[0]=='alu' and value in known and a in known and b in known
                and op[3]==[(known[value],1)]):
            recipes[target]=expression
    cfg['constant_expressions']=recipes
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/compact_914'))
    parser.add_argument('--output',type=Path,default=Path('results/overfetch_levels_914'))
    parser.add_argument('--trials',type=int,default=28)
    parser.add_argument('--iterations',type=int,default=650)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.source/'config.json').read_text())
    even=list(range(0,32,2));odd=list(range(1,32,2));all_groups=list(range(32))
    patterns=[(even,even,even),(even,all_groups,all_groups),(all_groups,all_groups,all_groups),
              (sorted(even+odd[::2]),all_groups,all_groups),
              (sorted(even+odd[::4]),sorted(even+odd[::2]),all_groups),
              ([],all_groups,all_groups)]
    jobs=[];screen=[]
    for pattern,levels in enumerate(patterns):
        for remaining,fill in ((12,False),(0,False),(0,True),(4,True)):
            raw=dict(base,overfetch_by_level={str(d):groups for d,groups in zip((8,9,10),levels)},
                     prefetch_madd_groups=base['prefetch_madd_groups'][:remaining])
            cfg=applicable_setup(raw,base)
            if fill:
                cfg.update(path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)])
                free=max(0,min(64,(874-counts(build(cfg))['engines']['flow'])//2))
                chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
                cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
                cfg=applicable_setup(cfg,base)
            cfg=balance(cfg,args.source,0)
            graph=build(cfg);bound=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
            index=len(jobs);directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
            row=dict(candidate=index,pattern=pattern,remaining_madds=remaining,fill_flow=fill,
                     resource_bound=bound['bound'],**counts(graph))
            screen.append(row)
            for name,value in (('config',cfg),('bounds',bound),('semantics',semantics),('parameters',row)):
                (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            jobs.append((index,cfg,args.source,args.output,args.trials,args.iterations))
            print(json.dumps(row),flush=True)
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
