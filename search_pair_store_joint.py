"""Combine early child rows, STORE spans and PC-offset arithmetic under 12k."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search,static_size
from search_dense_tables import configuration,SOURCE
from search_overfetch_levels import applicable_setup


def screen(job):
    index,parameters=job
    dense,number,keep,budget,madd,scalar_hash,target=parameters
    try:
        cfg=configuration((number,keep,budget,'tables'))
        cfg.update(dense_pc_tables=dense,pc_bit_pools=[] if dense else [2318],
                   early_pair_groups=list(range(0,32,4)),pair_even_odd_order=True,
                   all_even_odd_order=True,pc_offset_madd=madd,
                   scalar_constant_labels=['h6.a','h6.b'] if scalar_hash else [],
                   scalar_setup=['tree.shallow.bias','tree.d3.bias0'] if scalar_hash else [],
                   path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)])
        cfg=applicable_setup(cfg,cfg)
        free=max(0,min(64,(target-counts(build(cfg))['engines']['flow'])//2))
        chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
        cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
        cfg=applicable_setup(cfg,cfg);cfg=balance(cfg,SOURCE,0)
        graph=build(cfg);bounds=analyze_graph(graph)
        row=dict(candidate=index,parameters=parameters,resource_bound=bounds['bound'],
                 static_bundles_at_899=static_size(graph,899),**counts(graph))
        print(json.dumps(row),flush=True)
        return row,cfg,bounds
    except AssertionError as error:
        row=dict(candidate=index,parameters=parameters,rejected=str(error))
        print(json.dumps(row),flush=True)
        return row,None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/pair_store_joint_914'))
    parser.add_argument('--retain',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parameters=itertools.product((False,True),(6,11),((6,),(6,7)),(2,4),
                                 (False,True),(False,True),(871,874))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(screen,enumerate(parameters)))
    (args.output/'screen.json').write_text(json.dumps([r for r,_,_ in rows],indent=2)+'\n')
    valid=[row for row in rows if row[1] is not None and row[0]['static_bundles_at_899']<=11950]
    valid.sort(key=lambda row:(row[0]['resource_bound'],row[0]['weighted_alu_valu']))
    jobs=[]
    for row,cfg,bounds in valid[:args.retain]:
        graph=build(cfg);semantics=verify_semantics(graph,(0,1))
        directory=args.output/f"candidate_{row['candidate']:03}";directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bounds),('parameters',row),('semantics',semantics)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((row['candidate'],cfg,SOURCE,args.output,32,800))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
