"""Consume overfetch words earlier and spend selected freed FLOW slots."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build, counts, Scheduler, verify_semantics
from search_compact import search


def main():
    output=Path(__file__).resolve().parent
    jobs,screen=[],[]
    for index in (0,19,25,27,29):
        source=Path('results/overfetch_917')/f'candidate_{index:03}'
        config=json.loads((source/'config.json').read_text())
        options=[('mix_priority',dict(config,overfetch_mix_priority=True))]
        if index==19:
            old=config['prefetch_madd_groups']
            for count in (4,8,12,16):
                for end in ('head','tail'):
                    remaining=old[count:] if end=='head' else old[:-count]
                    options.append((f'madd_{count}_{end}',dict(config,prefetch_madd_groups=remaining,
                                                             overfetch_mix_priority=True)))
        for label,cfg in options:
            cfg=balance(cfg,source,0)
            g=build(cfg);Scheduler(g);verify_semantics(g,(0,1))
            candidate=len(jobs)
            screen.append(dict(candidate=candidate,source=str(source),label=label,**counts(g)))
            jobs.append((candidate,cfg,source,output,24,600))
    (output/'screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(search,jobs))
    (output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x[1]),indent=2)+'\n')
    print('RESULT',sorted(rows,key=lambda x:x[1]),flush=True)


if __name__=='__main__':
    main()
