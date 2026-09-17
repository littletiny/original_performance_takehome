"""Replace grandchild copies with packed rows and base-four dispatches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search,static_size
from search_overfetch_levels import applicable_setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/compact_914_pc'))
    parser.add_argument('--output',type=Path,default=Path('results/quad_dispatch_914'))
    parser.add_argument('--trials',type=int,default=24)
    parser.add_argument('--iterations',type=int,default=650)
    parser.add_argument('--retain',type=int,default=16)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.source/'config.json').read_text())
    profiles=[[[3,7]],[[2,3,7]],[[3,7],[10,11]],
              [[2,3,7],[10,11,14],[15,18]],
              [[2,3,7],[10,11,14],[15,18,22],[26,30]]]
    rows=[]
    for groups in profiles:
        members={k for group in groups for k in group}
        for keep in ((),(6,),(7,),(6,7)):
            for overfetch in (False,True):
                raw=dict(base,quad_dispatch_groups=groups,quad_row_buffers=1,
                         pair_even_odd_order=True,all_even_odd_order=True,
                         dense_pc_tables=len(members)>8,pc_bit_pools=[] if len(members)>8 else [2318],
                         heap_keep_levels=sorted({4,5,*keep}),
                         prefetch_madd_groups=[p for p in base['prefetch_madd_groups'] if p[1] not in members])
                if overfetch:raw['overfetch_by_level']={str(d):'all' for d in (8,9,10)}
                cfg=applicable_setup(raw,base);cfg=balance(cfg,args.source,0)
                graph=build(cfg);bounds=analyze_graph(graph);index=len(rows)
                directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
                row=dict(candidate=index,groups=groups,keep=list(keep),overfetch=overfetch,
                         resource_bound=bounds['bound'],static_bundles_at_899=static_size(graph,899),
                         **counts(graph))
                for name,value in (('config',cfg),('bounds',bounds),('parameters',row)):
                    (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
                rows.append((row,cfg));print(json.dumps(row),flush=True)
    (args.output/'screen.json').write_text(json.dumps([r for r,_ in rows],indent=2)+'\n')
    eligible=[item for item in rows if item[0]['static_bundles_at_899']<=11950]
    eligible.sort(key=lambda item:(item[0]['resource_bound'],item[0]['weighted_alu_valu']))
    selected=eligible[:args.retain]
    # Retain a small local control even when its work bound ranks below larger changes.
    for index in (0,1,8):
        if all(row['candidate']!=index for row,_ in selected):selected.append(rows[index])
    jobs=[]
    for row,cfg in selected:
        semantics=verify_semantics(build(cfg),(0,1))
        directory=args.output/f"candidate_{row['candidate']:03}"
        (directory/'semantics.json').write_text(json.dumps(semantics,indent=2)+'\n')
        jobs.append((row['candidate'],cfg,args.source,args.output,args.trials,args.iterations))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
