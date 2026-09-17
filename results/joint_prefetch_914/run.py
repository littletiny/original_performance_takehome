"""Test whether earlier depth-5 loads repair the combined graphs' LOAD window."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search,static_size
from search_overfetch_levels import applicable_setup
from search_pair_store_joint import screen,SOURCE


def main():
    output=Path(__file__).parent
    previous=json.loads(Path('results/pair_store_joint_914/screen.json').read_text())
    indexed={row['candidate']:row for row in previous}
    jobs=[];reports=[]
    for source_id in (3,111,123):
        _,base,_=screen((source_id,indexed[source_id]['parameters']))
        assert base is not None
        eligible=[k for k in range(32) if k not in base['prefetch5_groups']]
        for count in (0,4,8,16):
            groups=eligible[:count]
            cfg=dict(base,prefetch_pair_groups=groups,fold_path4_groups=groups,
                     path2_flow=True,path2_valu_groups=[(r,k) for r in (1,12) for k in range(32)])
            cfg=applicable_setup(cfg,base)
            free=max(0,min(64,(874-counts(build(cfg))['engines']['flow'])//2))
            chosen=[(r,k) for r in (12,1) for k in range(32)][:free]
            cfg['path2_valu_groups']=[(r,k) for r in (1,12) for k in range(32) if (r,k) not in chosen]
            cfg=applicable_setup(cfg,cfg);cfg=balance(cfg,SOURCE,0)
            graph=build(cfg);bounds=analyze_graph(graph);semantics=verify_semantics(graph,(0,1))
            index=len(reports)
            row=dict(candidate=index,source=source_id,groups=groups,resource_bound=bounds['bound'],
                     static_bundles_at_899=static_size(graph,899),**counts(graph))
            reports.append(row);directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
            for name,value in (('config',cfg),('bounds',bounds),('semantics',semantics),('parameters',row)):
                (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if bounds['bound']<=910 and row['static_bundles_at_899']<=11950 and count:
                jobs.append((index,cfg,SOURCE,output,24,600))
    (output/'screen.json').write_text(json.dumps(reports,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)


if __name__=='__main__':main()
