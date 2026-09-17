"""Joint STORE-copy spans, fixed PC pools, late-pair rows and LOAD budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, frozen, verify_semantics
from resource_bounds import analyze_graph
from search_compact import search, static_size


SOURCE=Path('results/compact_915')


def configuration(parameters):
    span,number,splits,keep,family,pools,chain_budget=parameters
    base=json.loads((SOURCE/'config.json').read_text())
    cfg=dict(base,overfetch_groups='all',prefetch_madd_groups=[],
             late_pair_groups=[0,4,8,12,16,20,24],late_pair_buffers=1,late_pair_order=[],
             dispatch_spans=[(3,k,span) for k in base['prefetch5_groups'][:number]],
             heap_keep_levels=[4,5]+list(keep),force_load_scalars=[],
             merge_chains=[],merge_regions=1,share_uniform_operands=True,
             scalar_root_groups='all',pc_bit_pools=[2318,2382,2446,2510][:pools],
             memory_vectors=[],memory_vector_order=[],lane_allocation_trials=16)
    cfg['dispatch_widths']=[x for x in base['dispatch_widths']
                            if x[0]!=3 or x[1] not in (0,4,8,12)[:splits]]
    graph=build(cfg)
    chains=[]
    for rnd in (3,14):
        pending=[];size=0
        for region in (r for r in graph.regions if r['round']==rnd):
            width=region.get('span',1)
            if size+width>chain_budget:
                if len(pending)>1:chains.append(pending)
                pending=[];size=0
            pending.append((rnd,region['groups'][0]));size+=width
        if len(pending)>1:chains.append(pending)
    cfg['merge_chains']=chains
    graph=build(cfg)
    free=max(0,min(64,(874-counts(graph)['engines']['flow'])//2))
    chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
    cfg.update(path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)
                                                if (r,k) not in chosen])
    graph=build(cfg)
    if family=='late':
        names=[name for name in base['memory_vectors'] if name in graph.names]
    else:
        c=[row[1] for row in frozen.HASH_STAGES]
        critical={f'broadcast.{x&0xffffffff}' for x in
                  (c[0],c[1],c[2]+c[3],c[2]<<9,c[4],c[5],4097,19,33,33<<9,9,16)}
        names=[name for name,op in zip(graph.names,graph.ops)
               if (op[1][0]=='vbroadcast' or name.startswith('derive.'))
               and name not in critical and name!='root.raw']
    order=sorted(names,key=lambda n:(n=='root.bias',not n.startswith('table.'),names.index(n)))
    cfg.update(memory_vectors=names,memory_vector_order=order)
    return balance(cfg,SOURCE,0)


def screen(job):
    index,parameters=job
    try:
        cfg=configuration(parameters)
        graph=build(cfg);bound=analyze_graph(graph)
        row=dict(candidate=index,parameters=parameters,resource_bound=bound['bound'],
                 static_bundles_at_899=static_size(graph,899),**counts(graph))
        print(json.dumps(row),flush=True)
        return row,cfg,bound
    except AssertionError as error:
        row=dict(candidate=index,parameters=parameters,rejected=str(error))
        print(json.dumps(row),flush=True)
        return row,None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/store_spans_915'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--retain',type=int,default=16)
    parser.add_argument('--trials',type=int,default=32)
    parser.add_argument('--iterations',type=int,default=600)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parameters=itertools.product((2,3),(6,11),(0,2,4),((7,),(6,7)),
                                 ('late','tables'),(1,4),(2,4))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(screen,enumerate(parameters)))
    (args.output/'screen.json').write_text(json.dumps([r for r,_,_ in rows],indent=2)+'\n')
    viable=[row for row in rows if row[1] is not None and row[0]['static_bundles_at_899']<=11950]
    viable.sort(key=lambda row:(row[0]['resource_bound'],row[0]['weighted_alu_valu']))
    chosen=viable[:args.retain]
    jobs=[]
    for row,cfg,bound in chosen:
        graph=build(cfg);semantics=verify_semantics(graph,(0,1))
        directory=args.output/f"candidate_{row['candidate']:03}";directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('parameters',row),('bounds',bound),('semantics',semantics)):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        jobs.append((row['candidate'],cfg,SOURCE,args.output,args.trials,args.iterations))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
