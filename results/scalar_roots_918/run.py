from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from optimize import build,counts,verify_semantics
from balance_resources import balance
from search_compact import search

if __name__=='__main__':
    source=Path('results/compact_918');output=Path('results/scalar_roots_918')
    base=json.loads((source/'config.json').read_text())
    groups=['all',[[0,k] for k in range(32)],[[11,k] for k in range(32)],[[0,0],[0,1]]]
    configs=[]
    for selected in groups:
        for header in (False,True):
            for policy in (0,2):
                cfg=balance(dict(base,header_root=header,scalar_root_groups=selected,precise_restore=True),source,policy)
                verify_semantics(build(cfg),(0,))
                configs.append(cfg)
    jobs=[(i,c,source,output,24,600) for i,c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)
