"""Reproduce three allocation failures without treating them as kernel scores."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import build,Scheduler,allocate
from search_compact import warm_keys,static_size


def main():
    output=Path(__file__).parent;rows=[]
    for index in (123,19,96):
        source=output/f'candidate_{index:03}'
        graph=build(json.loads((source/'config.json').read_text()));scheduler=Scheduler(graph)
        score,units,_=scheduler.search(warm_keys(ROOT/'results/compact_914_pc',graph,scheduler),250,999,.5)
        bases,audit=allocate(graph,scheduler.op_times(units))
        rows.append(dict(candidate=index,cycles_before_allocation=score,
                         static_bundles=static_size(graph,score),allocated=bases is not None,**audit))
    (output/'allocation_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
