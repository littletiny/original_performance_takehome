from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from optimize import build,verify_semantics
from search_compact import search

if __name__=='__main__':
    source=Path('results/compact_918');output=Path('results/header_root_918')
    base=json.loads((source/'config.json').read_text())
    configs=[dict(base,header_root=True,precise_restore=precise,heap_keep_levels=keep)
             for keep in ([5],[5,6]) for precise in (False,True)]
    for c in configs:verify_semantics(build(c),(0,1,901))
    jobs=[(i,c,source,output,48,800) for i,c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)
