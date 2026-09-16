from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from balance_resources import balance
from search_compact import search

if __name__ == '__main__':
    source=Path('results/compact_919')
    output=Path('results/pc_address_pools')
    config=dict(json.loads((source/'config.json').read_text()),pc_address_pools=True)
    configs=[config]+[balance(config,source,p) for p in range(7)]
    jobs=[(i,c,source,output,12,400) for i,c in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(results,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(results,key=lambda x:x[1]),flush=True)
