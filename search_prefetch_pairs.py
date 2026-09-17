"""Compare early depth-5 address prefixes and adjacent-pair scratch prefetch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, verify_semantics
from resource_bounds import analyze_graph
from search_compact import search, static_size


def configurations():
    production=Path('results/compact_915')
    base=json.loads((production/'config.json').read_text())
    eligible=[k for k in range(32) if k not in base['prefetch5_groups']]
    for mode in ('early_prefix_groups','prefetch_pair_groups'):
        for count in (2,4,8,len(eligible)):
            groups=eligible[:count]
            yield production,dict(base,fold_path4_groups=groups,**{mode:groups}),dict(
                family='local',mode=mode,groups=groups)
    source=Path('results/bound899_915/candidate_009')
    base=json.loads((source/'config.json').read_text())
    for mode in ('early_prefix_groups','prefetch_pair_groups'):
        for removed in (0,2):
            for keep in ([],[6,7]):
                grand=base['prefetch5_groups'][:-removed] if removed else base['prefetch5_groups']
                groups=[k for k in range(24) if k not in grand]
                cfg=dict(base,prefetch5_groups=grand,heap_keep_levels=[4,5]+keep,**{mode:groups})
                # Trade freed FLOW slots for shallow coordinate arithmetic.
                cfg.update(path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)])
                probe=build(dict(cfg,memory_vectors=[],memory_vector_order=[]))
                free=max(0,min(64,(874-counts(probe)['engines']['flow'])//2))
                chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
                cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
                yield source,cfg,dict(family='joint',mode=mode,groups=groups,removed=removed,keep=keep)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/prefetch_pairs_915'))
    parser.add_argument('--trials',type=int,default=24)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    jobs=[];screen=[]
    for index,(source,cfg,parameters) in enumerate(configurations()):
        probe=build(dict(cfg,memory_vectors=[],memory_vector_order=[]))
        available=set(probe.names)
        cfg['memory_vectors']=[name for name in cfg['memory_vectors'] if name in available]
        cfg['memory_vector_order']=[name for name in cfg['memory_vector_order'] if name in available]
        cfg=balance(cfg,source,0)
        graph=build(cfg);bound=analyze_graph(graph)
        row=dict(candidate=index,source=str(source),parameters=parameters,
                 resource_bound=bound['bound'],static_bundles_at_899=static_size(graph,899),**counts(graph))
        screen.append(row)
        directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (directory/'parameters.json').write_text(json.dumps(row,indent=2)+'\n')
        (directory/'bounds.json').write_text(json.dumps(bound,indent=2)+'\n')
        semantics=verify_semantics(graph,(0,1))
        (directory/'semantics.json').write_text(json.dumps(semantics,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        jobs.append((index,cfg,source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':
    main()
