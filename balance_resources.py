"""Choose an exact number of hash pack migrations from resource counts."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random
import re

import numpy as np

from optimize import build,counts,forced_scalar_pack


@lru_cache(maxsize=4)
def source_times(source):
    source=Path(source)
    graph=build(json.loads((source/'config.json').read_text()))
    times=np.load(source/'best.npz')['times']
    named=dict(zip(graph.names,map(int,times)))
    for name,t in list(named.items()):
        if '.lane' in name:
            stem=name.rsplit('.lane',1)[0]
            named[stem]=min(named.get(stem,t),t)
    regions={(r['round'],r['groups'][0]):int(times[r['start']]) for r in graph.regions}
    return named,regions


def balance(config,source,policy=0):
    graph=build(config)
    engines=counts(graph)['engines']
    a,v=engines['alu'],engines['valu']
    scalar=2*v>a
    approximate=abs(2*v-a)/10
    def cost(n):
        aa,vv=(a+8*n,v-n) if scalar else (a-8*n,v+n)
        return max((aa+11)//12,(vv+5)//6),abs(aa-2*vv),n
    number=min(range(max(0,int(approximate)-1),int(approximate)+3),key=cost)
    named,regions=source_times(str(source))
    centers=[regions[r,k]+4 for r,k,span in config.get('dispatch_spans',()) if span>1]
    if not centers:
        centers=[t+4 for t in regions.values()]
    if policy in ('pair_local','pair_final'):
        members={k for start in config.get('late_pair_groups',()) for k in (start,start+1)}
        local=[named[f'r{r}.g{k}.{label}'] for r,label in ((14,'bit'),(15,'mix'))
               for k in members if f'r{r}.g{k}.{label}' in named]
        if local:centers=local
    packs=defaultdict(list)
    pattern=re.compile(r'r\d+\.g\d+\.(?:mix|h2(?:\.[ab])?|h4|h6(?:\.[ab])?|bit)$')
    for i,name in enumerate(graph.names):
        stem=name.rsplit('.lane',1)[0]
        if pattern.fullmatch(stem): packs[stem].append(i)
    rng=random.Random(404+policy if isinstance(policy,int) else 404)
    candidates=[]
    for name,ids in packs.items():
        parts=name.split('.')
        if forced_scalar_pack(config,int(parts[0][1:]),int(parts[1][1:]),'.'.join(parts[2:])):
            continue
        engine=graph.ops[ids[0]][0]
        if scalar and engine!='valu' or not scalar and engine!='alu': continue
        assert len(ids)==(1 if scalar else 8)
        t=named.get(name,named.get('.'.join(name.split('.')[:2])+'.h1',1)-1)
        near=min(abs(t-c) for c in centers)
        rnd=int(parts[0][1:])
        if policy=='late':key=(t if scalar else -t,name)
        elif policy=='pair_local':key=(near,t,name)
        elif policy=='pair_final':key=(rnd!=15,not name.endswith(('h6.a','h6.b')),near,t,name)
        elif policy=='final_round':key=(rnd!=15,near,t,name)
        elif policy==0:key=(near,t,name)
        elif policy==1:key=(not name.endswith('h2.b'),near,t,name)
        else:key=(rng.random(),)
        candidates.append((key,name))
    assert len(candidates)>=number
    selected=[name for _,name in sorted(candidates)[:number]]
    changed=dict(config,scalar_overrides=dict(config.get('scalar_overrides') or {}))
    changed['scalar_overrides'].update({name:scalar for name in selected})
    check=counts(build(changed))
    expected=(a+8*number,v-number) if scalar else (a-8*number,v+number)
    assert (check['engines']['alu'],check['engines']['valu'])==expected
    return changed
