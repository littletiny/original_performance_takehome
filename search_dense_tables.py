"""Spend the space recovered by removing the fixed tail PC bank gap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,frozen,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search,static_size
from search_overfetch_levels import applicable_setup


SOURCE=Path('results/compact_914')


def configuration(parameters):
    number,keep,budget,family=parameters
    base=json.loads((SOURCE/'config.json').read_text())
    cfg=dict(base,dense_pc_tables=True,pc_bit_pools=[],constant_expressions=None,
             early_prefix_groups=[],fold_path4_groups=[],prefetch_madd_groups=[],
             overfetch_by_level={str(d):'all' for d in (8,9,10)},
             dispatch_spans=[(3,k,3) for k in base['prefetch5_groups'][:number]],
             heap_keep_levels=[4,5]+list(keep),force_load_scalars=[],
             merge_regions=1,merge_chains=[],late_pair_groups=[0,4,8,12,16,20,24],
             late_pair_buffers=1,late_pair_order=[],share_uniform_operands=True,
             scalar_root_groups='all',memory_vectors=[],memory_vector_order=[],
             path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)],
             lane_allocation_trials=16)
    graph=build(cfg);chains=[]
    for rnd in (3,14):
        pending=[];size=0
        for region in (r for r in graph.regions if r['round']==rnd):
            span=region.get('span',1)
            if size+span>budget:
                if len(pending)>1:chains.append(pending)
                pending=[];size=0
            pending.append((rnd,region['groups'][0]));size+=span
        if len(pending)>1:chains.append(pending)
    cfg['merge_chains']=chains
    free=max(0,min(64,(874-counts(build(cfg))['engines']['flow'])//2))
    chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
    cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
    graph=build(cfg)
    if family=='late':names=[n for n in base['memory_vectors'] if n in graph.names]
    else:
        c=[row[1] for row in frozen.HASH_STAGES]
        critical={f'broadcast.{x&0xffffffff}' for x in
                  (c[0],c[1],c[2]+c[3],c[2]<<9,c[4],c[5],4097,19,33,33<<9,9,16)}
        names=[name for name,op in zip(graph.names,graph.ops)
               if (op[1][0]=='vbroadcast' or name.startswith('derive.'))
               and name not in critical and name!='root.raw']
    cfg['memory_vectors']=names
    cfg['memory_vector_order']=sorted(names,key=lambda n:(n=='root.bias',not n.startswith('table.'),names.index(n)))
    return balance(cfg,SOURCE,0)


def screen(job):
    index,parameters=job
    try:
        cfg=configuration(parameters);graph=build(cfg);bounds=analyze_graph(graph)
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
    parser.add_argument('--output',type=Path,default=Path('results/dense_tables_914'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parameters=itertools.product((0,6,11),((6,),(7,),(6,7)),(2,4),('late','tables'))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(screen,enumerate(parameters)))
    (args.output/'screen.json').write_text(json.dumps([r for r,_,_ in rows],indent=2)+'\n')
    valid=[row for row in rows if row[1] is not None and row[0]['static_bundles_at_899']<=11950]
    valid.sort(key=lambda row:(row[0]['resource_bound'],row[0]['weighted_alu_valu']))
    base=json.loads((SOURCE/'config.json').read_text())
    control=applicable_setup(dict(base,dense_pc_tables=True,pc_bit_pools=[]),base)
    graph=build(control);bounds=analyze_graph(graph)
    selected=[(dict(candidate=100,parameters='dense layout only'),control,bounds),*valid[:8]]
    jobs=[]
    for row,cfg,bounds in selected:
        graph=build(cfg);semantics=verify_semantics(graph,(0,1))
        directory=args.output/f"candidate_{row['candidate']:03}";directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bounds),('parameters',row),('semantics',semantics)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((row['candidate'],cfg,SOURCE,args.output,32,700))
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
