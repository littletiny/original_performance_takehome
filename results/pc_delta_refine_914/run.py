"""Longer independent schedules of the lower-work, 910-bound graph."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from search_compact import search


if __name__=='__main__':
    source=Path('results/pc_delta_holes_914/candidate_000')
    config=json.loads((source/'config.json').read_text())
    output=Path(__file__).parent
    jobs=[(i,config,source,output,100,1200) for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    rows.sort(key=lambda row:row[1])
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESULT',rows,flush=True)
