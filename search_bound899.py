"""Schedule combined graphs only after their resource-window bound fits 899."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/bound899_915'))
    parser.add_argument('--trials',type=int,default=60)
    parser.add_argument('--iterations',type=int,default=800)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path('results/compact_915')
    base=json.loads(Path('results/cache_constant_trade_915/anchors_4/config.json').read_text())
    screen=[];accepted=[]
    for labels,roots,keep,anchors in itertools.product(
            (('h6.a','h6.b'),('h6.a',),()),(True,False),(False,True),(4,8,12,16,20)):
        cfg=dict(base,scalar_constant_labels=list(labels),scalar_root_groups='all' if roots else [],
                 heap_keep_levels=[4,5,6] if keep else [4,5],
                 force_load_scalars=[2310+8*k for k in range(2,2+anchors)],
                 memory_vectors=[],memory_vector_order=[])
        probe=build(cfg);available=set(probe.names)
        names=[name for name in base['memory_vectors'] if name in available]
        order=[name for name in base['memory_vector_order'] if name in available]
        if not roots and 'root.bias' not in names:
            names.append('root.bias');order.append('root.bias')
        cfg.update(memory_vectors=names,memory_vector_order=order)
        try:cfg=balance(cfg,source,0)
        except AssertionError as error:
            screen.append(dict(labels=labels,roots=roots,keep6=keep,anchors=anchors,rejected=str(error)))
            continue
        graph=build(cfg);bounds=analyze_graph(graph)
        row=dict(index=len(screen),labels=labels,roots=roots,keep6=keep,anchors=anchors,
                 resource_bound=bounds['bound'],**counts(graph))
        screen.append(row)
        if bounds['bound']>899:continue
        verify_semantics(graph,(0,1))
        index=len(accepted)
        directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (directory/'bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
        (directory/'parameters.json').write_text(json.dumps(row,indent=2)+'\n')
        row['candidate']=index
        accepted.append((index,cfg,source,args.output,args.trials,args.iterations))
        print(json.dumps(dict(candidate=index,**{k:row[k] for k in ('labels','roots','keep6','anchors','resource_bound','weighted_alu_valu')})),flush=True)
    (args.output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,accepted))
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
