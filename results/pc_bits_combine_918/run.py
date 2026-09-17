"""Combine the measured PC reduction with root and restoration reuse."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, verify_semantics
from search_compact import search


def main():
    source=Path('results/pc_bits_constant_918/candidate_000')
    output=Path(__file__).resolve().parent
    base=json.loads((source/'config.json').read_text())
    jobs,screen=[],[]
    for keep in ([5],[4,5]):
        for roots in ([],[[11,k] for k in range(32)],'all'):
            for policy in (0,1):
                cfg=dict(base,heap_keep_levels=keep,scalar_root_groups=roots,
                         header_root=True,precise_restore=True)
                cfg=balance(cfg,source,policy)
                graph=build(cfg)
                verify_semantics(graph,(0,1))
                index=len(jobs)
                screen.append(dict(candidate=index,keep=keep,roots=roots,policy=policy,**counts(graph)))
                jobs.append((index,cfg,source,output,16,500))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
