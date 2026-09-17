"""Re-evaluate exact restore ranges and larger temporary buffer counts."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from optimize import build,verify_semantics
from search_compact import search

if __name__=='__main__':
    output=Path('results/precise_restore_918')
    jobs=[]
    for source in (Path('results/compact_918'),Path('results/depth4_918/candidate_027')):
        base=json.loads((source/'config.json').read_text())
        for keep,buffers in (([5],4),([5,6],4),([5,6],6),([5,6],8),([5,7],4),([4,5,6,7],4),([4,5,6,7],6)):
            cfg=dict(base,precise_restore=True,heap_keep_levels=keep,temp_buffers=buffers,lane_allocation_trials=32)
            verify_semantics(build(cfg),(0,))
            jobs.append((len(jobs),cfg,source,output,32,600))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(results,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(results,key=lambda x:x[1]),flush=True)
