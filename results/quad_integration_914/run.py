"""Frozen-machine control for quartet rows, child rows and chained dispatches."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from balance_resources import balance
from optimize import build,counts,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    source=ROOT/'results/quad_natural_914/candidate_015';output=Path(__file__).parent
    base=json.loads((source/'config.json').read_text())
    raw=dict(base,early_pair_groups=[24,28],late_pair_groups=[20,24],
             late_pair_order=[24,20],quad_row_buffers=2,
             merge_chains=[[[3,0],[3,4]],[[3,10],[3,11]]])
    cfg=applicable_setup(raw,base);cfg=balance(cfg,source,0)
    graph=build(cfg);directory=output/'candidate_000';directory.mkdir(exist_ok=True)
    for name,value in (('config',cfg),('bounds',analyze_graph(graph)),
                       ('semantics',verify_semantics(graph,(0,1)))):
        (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(counts(graph)),flush=True)
    result=search((0,cfg,source,output,32,750))
    (output/'summary.json').write_text(json.dumps(result)+'\n');print('RESULT',result,flush=True)


if __name__=='__main__':main()
