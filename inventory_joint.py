"""Screen combined work reductions by fixed-graph bounds before scheduling."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, frozen, verify_semantics
from resource_bounds import analyze_graph
from search_compact import static_size


SOURCE=Path('results/compact_915')


def configuration(parameters):
    fold,family,labels,roots,flow_target=parameters
    base=json.loads((SOURCE/'config.json').read_text())
    cfg=dict(base,overfetch_groups='all',prefetch_madd_groups=[],
             fold_path4_groups=list(range(fold)),merge_chains=[],merge_regions=4,
             dispatch_widths=base['dispatch_widths']+[[14,2,2],[14,6,2]],
             late_pair_groups=[0,2,4,6,8,12,16,20,24],late_pair_buffers=1,late_pair_order=[],
             memory_vectors=[],memory_vector_order=[],memory_vector_buffers=1,
             share_uniform_operands=True,scalar_constant_labels=list(labels),
             scalar_setup=['tree.shallow.bias','tree.d3.bias0'] if 'h6.a' in labels else [],
             scalar_root_groups='all' if roots else [])
    graph=build(cfg)
    free=max(0,min(64,(flow_target-counts(graph)['engines']['flow'])//2))
    selected=[(r,k) for r in (12,1) for k in range(32)][:free]
    cfg.update(path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)
                                                if (r,k) not in selected])
    graph=build(cfg)
    c=[row[1] for row in frozen.HASH_STAGES]
    critical={f'broadcast.{x&0xffffffff}' for x in
              (c[0],c[1],c[2]+c[3],c[2]<<9,c[4],c[5],4097,19,33,33<<9,9,16)}
    first={f'broadcast.{c[0]}','broadcast.4097','root.raw'}
    names=[]
    for name,op in zip(graph.names,graph.ops):
        if op[1][0]!='vbroadcast' and not name.startswith('derive.'):continue
        if family=='late' and (name in critical or name=='root.raw' or name.startswith('table.')):continue
        if family=='tables' and (name in critical or name=='root.raw'):continue
        if family=='keep_h1' and name in first:continue
        names.append(name)
    # Scalars underlying these vectors are independent of the vector results.
    # Keep early table operands ahead of the post-wrap root and late constants.
    order=sorted(names,key=lambda n:(n=='root.bias',not n.startswith('table.'),names.index(n)))
    cfg.update(memory_vectors=names,memory_vector_order=order)
    return balance(cfg,SOURCE,0),free


def run(job):
    index,parameters=job
    fold,family,labels,roots,flow_target=parameters
    report=dict(candidate=index,fold=fold,memory_family=family,scalar_labels=list(labels),
                scalar_roots=roots,flow_target=flow_target)
    try:
        cfg,path2=configuration(parameters)
        graph=build(cfg)
        bound=analyze_graph(graph)
        accounting=counts(graph)
        count_bound=accounting.pop('bound')
        report.update(status='bounded',path2_groups=path2,bound=bound['bound'],
                      count_bound=count_bound,**accounting,resource_windows=bound['resource_windows'],
                      critical_path_bound=bound['critical_path_bound'],
                      graph_sha256=bound['graph_sha256'],
                      static_bundles_at_899=static_size(graph,899))
        print(json.dumps({k:report[k] for k in ('candidate','bound','weighted_alu_valu','static_bundles_at_899')}),flush=True)
        return report,cfg
    except AssertionError as error:
        report.update(status='unsupported_configuration',error=str(error))
        print(json.dumps(report),flush=True)
        return report,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/joint_inventory_915'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--retain',type=int,default=12)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    options=itertools.product((24,32),('late','tables','keep_h1','all'),
                              ((),('h2.a','h2.b'),('h6.a','h6.b'),('h2.a','h6.a'),('h2.b','h6.a')),
                              (False,True),(867,875,877))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(run,enumerate(options)))
    viable=[(r,cfg) for r,cfg in rows if cfg is not None and r['static_bundles_at_899']<=12000]
    viable.sort(key=lambda x:(x[0]['bound'],x[0]['weighted_alu_valu']))
    retained=[]
    for report,cfg in viable[:args.retain]:
        graph=build(cfg)
        semantics=verify_semantics(graph,(0,1))
        directory=args.output/f"candidate_{report['candidate']:03}"
        directory.mkdir(exist_ok=True)
        (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (directory/'bounds.json').write_text(json.dumps(report,indent=2)+'\n')
        (directory/'semantics.json').write_text(json.dumps(semantics,indent=2)+'\n')
        retained.append(report['candidate'])
    summary=dict(candidates=len(rows),retained=retained,
                 necessary_conditions_passed=[r['candidate'] for r,cfg in viable if r['bound']<900],
                 rows=sorted([r for r,cfg in rows],key=lambda r:(r.get('bound',100000),r.get('weighted_alu_valu',100000))))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
