"""Reuse contiguous temporary-buffer addresses as interleaved dispatch PCs."""
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
    parser.add_argument('--output',type=Path,default=Path('results/interleaved_pc_914'))
    parser.add_argument('--trials',type=int,default=32)
    parser.add_argument('--iterations',type=int,default=750)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads((args.source/'config.json').read_text());original=build(base)
    singles=[(r['round'],r['groups'][0]) for r in original.regions if r['width']==1]
    jobs=[];rows=[]
    for number in (8,16,20):
        for position in ('front','back'):
            groups=singles[4:number]+singles[-4:] if position=='front' else singles[-number:]
            assert len(groups)==number and len(set(groups))==number
            for removed in (0,4,8,12):
                raw=dict(base,dense_pc_tables=True,pc_bit_pools=[],pc_interleave_groups=groups,
                         prefetch_madd_groups=base['prefetch_madd_groups'][:12-removed])
                probe=build(dict(raw,memory_vectors=[],memory_vector_order=[],constant_expressions=None))
                additions=[]
                for value in [*(row['delta'] for row in probe.pc_madd_offsets),8*number]:
                    name=next((name for name in (f'broadcast.{value}',f'derive.{value}') if name in probe.names),None)
                    if name is not None and name not in base['memory_vectors']:additions.append(name)
                setup=dict(base,memory_vectors=base['memory_vectors']+additions,
                           memory_vector_order=additions+base['memory_vector_order'])
                cfg=applicable_setup(raw,setup);cfg=balance(cfg,args.source,0)
                graph=build(cfg);bounds=analyze_graph(graph);index=len(jobs)
                row=dict(candidate=index,number=number,position=position,removed_madds=removed,
                         memory_additions=additions,resource_bound=bounds['bound'],
                         static_bundles_at_899=static_size(graph,899),**counts(graph))
                directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
                semantics=verify_semantics(graph,(0,1))
                for name,value in (('config',cfg),('bounds',bounds),('parameters',row),('semantics',semantics)):
                    (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
                rows.append(row);print(json.dumps(row),flush=True)
                jobs.append((index,cfg,args.source,args.output,args.trials,args.iterations))
    (args.output/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1]);(args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
