"""Longer independent searches of the final reduced-work 913 graph."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from search_compact import search


if __name__=='__main__':
    source=ROOT/'results/load_holes_913/candidate_002';output=Path(__file__).parent
    cfg=json.loads((source/'config.json').read_text())
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,[(i,cfg,source,output,80,1000) for i in range(101,105)]))
    rows.sort(key=lambda row:row[1]);(output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)
