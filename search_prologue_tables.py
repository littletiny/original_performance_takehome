"""Spend early FLOW slots on bias constants, then materialize shallow tables."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from balance_resources import balance
from optimize import build,counts,frozen,verify_semantics
from resource_bounds import analyze_graph
from search_compact import search
from search_overfetch_levels import applicable_setup


def main():
    source=Path('results/interleaved_pc_914/candidate_001')
    output=Path('results/prologue_tables_914');output.mkdir(exist_ok=True)
    base=json.loads((source/'config.json').read_text());c6=frozen.HASH_STAGES[-1][1]
    profiles={'d1':['table.d1.n0','table.d1.n1'],
              'd2':[f'table.d2.n{i}' for i in range(4)],
              'd1_half_d2':['table.d1.n0','table.d1.n1','table.d2.n0','table.d2.n1'],
              'all':['table.d1.n0','table.d1.n1',*[f'table.d2.n{i}' for i in range(4)]]}
    jobs=[];rows=[]
    for family,tables in profiles.items():
        for derive_two in (False,True):
            for scalar_roots in (False,True):
                constants=[c6,2318]+([2] if derive_two else [])
                priority={f'constant.{c6}':-8,'constant.2318':-7,'initial.pause':-6,
                          'tree.d0.raw0':-5,f'broadcast.{c6}':-6,'tree.shallow.bias':-4}
                if derive_two:priority['constant.2']=-5
                raw=dict(base,pc_prologue=13,flow_constants=constants,warm_priorities=priority,
                         scalar_root_groups='all' if scalar_roots else [])
                selected=tables+(['derive.2'] if derive_two else [])
                setup=dict(base,memory_vectors=base['memory_vectors']+selected,
                           memory_vector_order=selected+base['memory_vector_order'])
                cfg=applicable_setup(raw,setup);cfg=balance(cfg,source,0)
                graph=build(cfg);bounds=analyze_graph(graph);index=len(jobs)
                row=dict(candidate=index,family=family,derive_two=derive_two,scalar_roots=scalar_roots,
                         resource_bound=bounds['bound'],**counts(graph))
                directory=output/f'candidate_{index:03}';directory.mkdir(exist_ok=True)
                for name,value in (('config',cfg),('bounds',bounds),('parameters',row),
                                   ('semantics',verify_semantics(graph,(0,1)))):
                    (directory/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
                rows.append(row);print(json.dumps(row),flush=True)
                jobs.append((index,cfg,source,output,40,850))
    (output/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(search,jobs))
    results.sort(key=lambda row:row[1]);(output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',results,flush=True)


if __name__=='__main__':main()
