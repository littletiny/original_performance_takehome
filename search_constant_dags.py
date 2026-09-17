"""Schedule exact scalar-constant resynthesis on production and control graphs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/constant_dags_915'))
    parser.add_argument('--trials',type=int,default=40)
    parser.add_argument('--iterations',type=int,default=800)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sources=['results/compact_915','results/prefetch_pairs_915/candidate_000',
             'results/prefetch_pairs_915/candidate_006','results/bound899_915/candidate_009',
             'results/store_spans_915/candidate_164','results/store_spans_915/candidate_172']
    jobs=[]
    for index,relative in enumerate(sources):
        source=Path(relative)
        base=json.loads((source/'config.json').read_text())
        warm=source if (source/'best.npz').exists() else Path('results/compact_915')
        cfg=balance(dict(base,resynthesize_constants=True),warm,0)
        graph=build(cfg);bounds=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
        directory=args.output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
        for name,value in (('config',cfg),('bounds',bounds),('semantics',semantics),
                           ('rewrites',graph.constant_rewrites),('source',dict(source=str(source),warm=str(warm)))):
            (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(dict(candidate=index,source=relative,resource_bound=bounds['bound'],
                              rewrites=len(graph.constant_rewrites),**bounds['counts'])),flush=True)
        jobs.append((index,cfg,warm,args.output,args.trials,args.iterations))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
