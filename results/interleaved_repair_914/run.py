"""Try one-cycle repairs of the reduced-work interleaved-PC graph."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from repair_schedule import repair


def run(window):
    return repair(ROOT/'results/interleaved_pc_914/candidate_001',
                  Path(__file__).parent/f'window_{window}',913,window,50,2,True)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,(1,2,4,8)))
    (Path(__file__).parent/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
