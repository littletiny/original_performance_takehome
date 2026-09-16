"""Repeat the two bounded follow-up searches recorded in summary.json."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from search_compact import search

if __name__ == '__main__':
    sources=[Path('results/path_choices_918/candidate_015'),
             Path('results/path_choices_918/candidate_011')]
    output=Path('results/path_refine_918')
    jobs=[(i,json.loads((source/'config.json').read_text()),source,output,60,800)
          for i,source in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(rows,flush=True)
