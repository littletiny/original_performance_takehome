"""Bind the promoted 913-cycle program, accounting and validation artifacts."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import bootstrap_cycle,build,counts
from plot_utilization import profile
from resource_bounds import analyze


def main():
    source=Path(__file__).parent
    graph=build(json.loads((source/'config.json').read_text()))
    times=np.load(source/'best.npz')['times'];bootstrap=bootstrap_cycle(graph,times)
    bound=analyze(source);(source/'bounds.json').write_text(json.dumps(bound,indent=2)+'\n')
    classify=runpy.run_path(str(ROOT/'results/work_inventory_918/run.py'))['classify']
    categories=defaultdict(Counter)
    for name,op in zip(graph.names,graph.ops):
        if op[0] not in ('alu','valu'):continue
        category,detail=classify(name.replace('.pair_address','.address'),op)
        categories[category][detail]+=8 if op[0]=='valu' else 1
    accounting=counts(graph)
    assert sum(sum(row.values()) for row in categories.values())==accounting['weighted_alu_valu']==54406
    accounting['static_padding_upper_bound']=accounting['static_padding_bundles']
    accounting['static_padding_bundles']-=bootstrap
    accounting.update(cycles=913,bootstrap_cycle=bootstrap,static_main_bundles=913-8*len(graph.regions),
                      bound_scope='Counts for this fixed graph; not a global task lower bound.',
                      work_categories_W={key:dict(total=sum(value.values()),details=dict(value))
                                         for key,value in categories.items()})
    assert accounting['static_main_bundles']==553 and accounting['static_padding_bundles']==384
    (source/'analysis.json').write_text(json.dumps(accounting,indent=2)+'\n')
    current=(ROOT/'perf_takehome.py').read_text()
    previous=subprocess.check_output(['git','show','48ce255:perf_takehome.py'],cwd=ROOT,text=True)
    def outside(text):
        before,rest=text.split('# SUB900_CANDIDATE_BEGIN\n',1)
        _,after=rest.split('# SUB900_CANDIDATE_END\n',1)
        return before,after.lstrip('\n')
    assert outside(current)==outside(previous)
    digest=hashlib.sha256(current.encode()).hexdigest()
    primary=json.loads((source/'verification.json').read_text())
    extra=json.loads((source/'verification_extra.json').read_text())
    initial=json.loads((source/'port_utilization.json').read_text())
    assert extra['source_sha256']==initial['source_sha256']==digest
    assert primary['cycles']==extra['cycles']==initial['cycles']==913
    rows0=np.loadtxt(source/'port_utilization.csv',delimiter=',',skiprows=1,dtype=np.int64)
    rows1,second=profile(1)
    assert np.array_equal(rows0[:,2:],rows1[:,2:])
    assert not np.array_equal(rows0[:,1],rows1[:,1])
    (source/'port_utilization_seed1.json').write_text(json.dumps(second,indent=2)+'\n')
    for engine,count in accounting['engines'].items():
        assert initial['ports'][engine]['used_slots']==count+int(engine=='flow')
    tests={}
    for name,expected in (('submission',9),('native',3),('regression',38)):
        log=(source/f'{name}_tests.log').read_text()
        count=int(re.search(r'Ran (\d+) tests?',log)[1])
        assert count==expected and '\nOK\n' in log
        tests[f'{name}_tests_passed']=count
    report=dict(source_sha256=digest,production_changes_confined_to_generated_block=True,
                cycles=913,static_bundles=primary['static_bundles'],
                static_slot_operations=primary['static_slots'],weighted_arithmetic=54406,
                previous_914_weighted_arithmetic=54484,**tests,
                full_checkpoint_seeds=primary['seeds']+extra['seeds'],checkpoints_per_seed=20480,
                source_bound_port_profile=True,seed0_seed1_identical_per_cycle_ports=True,
                seed0_seed1_different_pc_paths=True,strict_sub900_achieved=False,
                static_bundle_cap=12000,fixed_graph_lower_bound=bound['bound'],
                strict_sub900_counts_only_remaining_reduction=54406-60*899,
                bootstrap_cycle=bootstrap,initial_pause_in_graph=True,
                scratch_live_peak=primary['lane_live_peak'],static_bundles_saved_vs_unpacked_913=512)
    assert report['static_bundles']==10537 and bound['bound']==909
    (source/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
