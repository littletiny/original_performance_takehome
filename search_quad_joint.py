"""Combine packed grandchild dispatches with child rows and earlier setup."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from functools import lru_cache
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,frozen,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search,static_size
from search_dense_tables import configuration


SOURCE=Path('results/compact_914_pc')


@lru_cache(maxsize=16)
def source_config(keep,budget):
    return configuration((0,keep,budget,'late'))


def screen(job):
    index,parameters=job
    number,keep,budget,row_buffers,memory_buffers,family,target=parameters
    cfg=copy.deepcopy(source_config(tuple(keep),budget))
    groups=([[2,3,7],[10,11,14],[15,18]] if number==8 else
            [[2,3,7],[10,11,14],[15,18,22],[26,30]])
    cfg.update(quad_dispatch_groups=groups,quad_row_buffers=row_buffers,
               early_pair_groups=list(range(0,32,4)),pair_even_odd_order=True,
               all_even_odd_order=True,pc_offset_madd=True,
               scalar_constant_labels=['h6.a','h6.b'],
               scalar_setup=['tree.shallow.bias','tree.d3.bias0'],
               memory_vectors=[],memory_vector_order=[],memory_vector_buffers=memory_buffers,
               path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)])
    try:
        graph=build(cfg)
        constants=[stage[1] for stage in frozen.HASH_STAGES]
        critical={f'broadcast.{x&0xffffffff}' for x in
                  (constants[0],constants[1],constants[2]+constants[3],constants[2]<<9,
                   constants[4],constants[5],4097,19,33,33<<9,9,16)}
        names=[name for name,op in zip(graph.names,graph.ops)
               if (op[1][0]=='vbroadcast' or name.startswith('derive.'))
               and name not in critical and name!='root.raw'
               and not (family=='keep_first' and name.startswith('table.d1.'))]
        cfg['memory_vectors']=names
        cfg['memory_vector_order']=sorted(names,key=lambda n:(n=='root.bias',not n.startswith('table.'),names.index(n)))
        free=max(0,min(64,(target-counts(build(cfg))['engines']['flow'])//2))
        chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
        cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
        cfg=balance(cfg,SOURCE,0);graph=build(cfg);bounds=analyze_graph(graph)
        row=dict(candidate=index,parameters=parameters,resource_bound=bounds['bound'],
                 static_bundles_at_899=static_size(graph,899),**counts(graph))
        return row,cfg,bounds
    except AssertionError as error:
        return dict(candidate=index,parameters=parameters,rejected=str(error)),None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/quad_joint_914'))
    parser.add_argument('--retain',type=int,default=12)
    parser.add_argument('--trials',type=int,default=24)
    parser.add_argument('--iterations',type=int,default=700)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parameters=itertools.product((8,11),((7,),(6,7)),(1,2,4),(1,2),(1,4),
                                 ('all','keep_first'),(868,874))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row,cfg,bounds in pool.map(screen,enumerate(parameters)):
            rows.append((row,cfg,bounds));print(json.dumps(row),flush=True)
    (args.output/'screen.json').write_text(json.dumps([row for row,_,_ in rows],indent=2)+'\n')
    valid=[r for r in rows if r[1] is not None and r[0]['static_bundles_at_899']<=11920]
    valid.sort(key=lambda r:(r[0]['resource_bound'],r[0]['weighted_alu_valu']))
    jobs=[]
    for row,cfg,bounds in valid[:args.retain]:
        graph=build(cfg);semantics=verify_semantics(graph,(0,1))
        directory=args.output/f"candidate_{row['candidate']:03}";directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bounds),('parameters',row),('semantics',semantics)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((row['candidate'],cfg,SOURCE,args.output,args.trials,args.iterations))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
